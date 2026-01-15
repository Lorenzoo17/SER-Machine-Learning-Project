import os
from glob import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import soundfile as sf
import matplotlib.pyplot as plt

# =========================
# LABELING RAVDESS (emotion)
# =========================
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# mapping a interi 0..7 (comodo per CrossEntropyLoss)
EMOTION_TO_INDEX = {name: i for i, name in enumerate(EMOTION_MAP.values())}


def parse_ravdess_filename(filepath: str) -> Dict[str, str]:
    """
    Estrae i campi principali dal filename RAVDESS.
    Esempio: 03-01-01-01-01-01-01.wav
    """
    filename = os.path.basename(filepath)
    parts = filename.split("-")
    # parts: [modality, vocal_channel, emotion, intensity, statement, repetition, actor.wav]
    actor = parts[-1].split(".")[0]
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion_id": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": actor,
    }


def extract_emotion_label(filepath: str) -> str:
    """Ritorna label testuale (es. 'happy')."""
    info = parse_ravdess_filename(filepath)
    return EMOTION_MAP[info["emotion_id"]]


def extract_emotion_index(filepath: str) -> int:
    """Ritorna label numerica 0..7."""
    label = extract_emotion_label(filepath)
    return EMOTION_TO_INDEX[label]


def list_ravdess_files(data_root: str) -> List[str]:
    """
    Cerca tutti i wav nella struttura:
    data_root/Actor_XX/*.wav
    """
    return glob(os.path.join(data_root, "Actor_*", "*.wav"))


def split_by_speakers(
    filepaths: List[str],
    train_speakers: List[str],
    val_speakers: List[str],
    test_speakers: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split speaker-independent: assegna i file agli split in base all'attore.
    Nota: speaker id sono stringhe tipo "01", "02", ... "24".
    """
    train, val, test = [], [], []
    for fp in filepaths:
        actor = parse_ravdess_filename(fp)["actor"]
        if actor in train_speakers:
            train.append(fp)
        elif actor in val_speakers:
            val.append(fp)
        elif actor in test_speakers:
            test.append(fp)
        # se non è in nessuna lista, lo ignoriamo
    return train, val, test

def filter_audio_speech(filepaths: List[str]) -> List[str]:
    """Tiene solo audio-only (03) e speech (01)."""
    out = []
    for fp in filepaths:
        info = parse_ravdess_filename(fp)
        if info["modality"] == "03" and info["vocal_channel"] == "01":
            out.append(fp)
    return out

def plot_log_mel_from_loader(data_loader, idx: int = 0):
    """
    Stampa il log-Mel spectrogram di un singolo campione dal DataLoader.
    
    Args:
        data_loader: PyTorch DataLoader
        idx: indice del campione nel batch (default 0)
    """
    X, y = next(iter(data_loader))
    
    mel = X[idx].squeeze(0)   # [64, 401]
    label = y[idx].item()

    plt.figure(figsize=(10, 4))
    plt.imshow(
        mel.numpy(),
        origin="lower",
        aspect="auto"
    )
    plt.colorbar()
    plt.title(f"Log-Mel Spectrogram - label {label}")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.show()
    
def trim_silence_np(audio: np.ndarray, sr: int, top_db: float = 30.0) -> np.ndarray:
    """
    Silence trimming semplice (inizio/fine) basato su energia.
    - top_db più basso = taglia meno
    - top_db più alto = taglia di più
    """
    # RMS (energia) su finestre
    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms

    if len(audio) < frame_length:
        return audio

    # Calcolo RMS per frame
    rms = []
    for i in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[i:i + frame_length]
        rms.append(np.sqrt(np.mean(frame**2) + 1e-12))
    rms = np.array(rms)

    # Soglia: max_rms / (10^(top_db/20))
    max_rms = rms.max()
    thr = max_rms / (10 ** (top_db / 20))

    # Trova i frame "attivi"
    idx = np.where(rms >= thr)[0]
    if len(idx) == 0:
        return audio  # fallback: non tagliare

    start_frame = idx[0]
    end_frame = idx[-1]

    start_sample = start_frame * hop_length
    end_sample = min(len(audio), end_frame * hop_length + frame_length)

    return audio[start_sample:end_sample]

def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform)
    return waveform + noise_level * noise

def random_gain(waveform, min_gain=0.8, max_gain=1.2):
    gain = torch.empty(1).uniform_(min_gain, max_gain).item()
    return waveform * gain

def time_shift(waveform, max_shift=0.1):
    shift = int(waveform.shape[1] * max_shift)
    shift = np.random.randint(-shift, shift)
    return torch.roll(waveform, shifts=shift, dims=1)


# DATASET PYTORCH
class RavdessDataset(Dataset):
    """
    Dataset che ritorna:
      X: torch.Tensor [1, 64, 401]  (log-mel in dB)
      y: torch.LongTensor scalar     (0..7)
    """

    def __init__(
        self,
        filepaths: List[str],
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        max_duration: float = 4.0,
        use_db: bool = True,
        top_db: Optional[float] = 80.0,
        augmentation: bool = False,
    ):
        self.filepaths = filepaths
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.augmentation = augmentation

        # Trasformazioni (torchaudio)
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,  # power spectrogram
        )

        self.use_db = use_db
        self.to_db = None
        if use_db:
            # stype="power" coerente con power=2.0 sopra
            self.to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

        # resampler creato "on demand" quando serve
        self._resamplers: Dict[int, T.Resample] = {}

        # =========================
        # AUGMENTATION PARAMS
        # =========================
        self.p_noise = 1
        self.p_gain = 0.5
        self.p_shift = 0.5
        self.p_sox = 0.6          # pitch + speed (se sox disponibile)
        self.p_specaug = 0.7

        # Noise SNR range (dB)
        self.snr_min_db = 10.0
        self.snr_max_db = 30.0

        # Gain range
        self.min_gain = 0.7
        self.max_gain = 1.3

        # Time shift max fraction of length
        self.max_shift = 0.10

        # SpecAugment (su log-mel)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.num_freq_masks = 2
        self.num_time_masks = 2


    def _add_noise_snr(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, T]
        snr_db = float(np.random.uniform(self.snr_min_db, self.snr_max_db))
        sig_power = waveform.pow(2).mean().clamp_min(1e-12)

        noise = torch.randn_like(waveform)
        noise_power = noise.pow(2).mean().clamp_min(1e-12)

        # scala rumore per ottenere SNR desiderato
        scale = torch.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10))))
        return waveform + noise * scale
    
    def _add_noise_paper(self, waveform: torch.Tensor, noise_ratio: float = 0.005) -> torch.Tensor:
        """
        Noise augmentation come nel paper:
        - Rumore gaussiano N(0,1)
        - Ampiezza rumore = noise_ratio * max(|signal|)
        """
        # waveform: [1, T]
        max_amp = waveform.abs().max().clamp_min(1e-12)  # evita 0
        noise_std = noise_ratio * max_amp

        noise = torch.randn_like(waveform) * noise_std
        return waveform + noise

    def _random_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        g = float(np.random.uniform(self.min_gain, self.max_gain))
        return waveform * g

    def _time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        # shift su asse temporale
        max_samp = int(waveform.shape[1] * self.max_shift)
        if max_samp <= 0:
            return waveform
        shift = int(np.random.randint(-max_samp, max_samp + 1))
        return torch.roll(waveform, shifts=shift, dims=1)

    def _sox_pitch_speed(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Pitch shift + speed perturbation tramite sox (se disponibile).
        - pitch in cents (±200 ~ ±2 semitoni)
        - speed (0.9–1.1) e poi resample a sr per mantenere sample rate costante
        """
        try:
            import torchaudio
            from torchaudio.sox_effects import apply_effects_tensor

            pitch_cents = int(np.random.uniform(-200, 200))
            speed = float(np.random.uniform(0.9, 1.1))

            effects = [
                ["pitch", str(pitch_cents)],
                ["speed", f"{speed:.4f}"],
                ["rate", str(sr)],  # torna al sample rate target
            ]
            wav_out, _ = apply_effects_tensor(waveform, sr, effects)
            return wav_out
        except Exception:
            # se sox/sox_effects non è disponibile, salta senza rompere tutto
            return waveform

    def _augment_waveform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        # ordine ragionevole: gain/noise/shift/sox, poi clamp
        '''
        if np.random.rand() < self.p_gain:
           waveform = self._random_gain(waveform)

        if np.random.rand() < self.p_noise:
            waveform = self._add_noise_snr(waveform)

        if np.random.rand() < self.p_shift:
            waveform = self._time_shift(waveform)

        if np.random.rand() < self.p_sox:
            waveform = self._sox_pitch_speed(waveform, sr)
    '''
        # Noise augmentation (paper)
        if np.random.rand() < self.p_noise:
            waveform = self._add_noise_paper(waveform)

        return waveform.clamp(-1.0, 1.0)

    def _spec_augment(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [1, n_mels, time] (log-mel)
        Applica SpecAugment (time mask + freq mask) più volte.
        """
        x = feat
        for _ in range(self.num_freq_masks):
            x = self.freq_mask(x)
        for _ in range(self.num_time_masks):
            x = self.time_mask(x)
        return x


    def __len__(self) -> int:
        return len(self.filepaths)

    def _get_resampler(self, orig_sr: int) -> T.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
        return self._resamplers[orig_sr]

    def _load_waveform(self, path: str) -> Tuple[torch.Tensor, int]:
        # soundfile -> numpy float32
        audio, sr = sf.read(path, dtype="float32")

        # stereo -> mono perche non ci servono le informazioni di spazio e ambiente
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # silence trimming (prima del resample e del pad/crop)
        audio = trim_silence_np(audio, sr, top_db=30.0)

        waveform = torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]
        return waveform, sr

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        # pad/crop a lunghezza fissa
        num_samples = waveform.shape[1]
        if num_samples < self.max_samples:
            pad = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, : self.max_samples]
        return waveform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.filepaths[idx]

        # y come intero 0..7
        y = torch.tensor(extract_emotion_index(path), dtype=torch.long)

        # carico audio
        waveform, sr = self._load_waveform(path)

        # resample se serve
        if sr != self.sample_rate:
            resampler = self._get_resampler(sr)
            waveform = resampler(waveform)

        # durata fissa
        waveform = self._fix_length(waveform)

        if self.augmentation:
            waveform = self._augment_waveform(waveform, self.sample_rate)



        mel_spec = self.mel(waveform)

        if self.use_db and self.to_db is not None:
            feat = self.to_db(mel_spec)
        else:
            feat = torch.log(mel_spec + 1e-9)

        if self.augmentation and (np.random.rand() < self.p_specaug):
            feat = self._spec_augment(feat)

        return feat, y


 
