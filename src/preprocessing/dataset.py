import os
from glob import glob
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

import torchaudio.transforms as T
import torchaudio.functional as F

import soundfile as sf


EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


EMOTIONS = list(EMOTION_MAP.values())
# Dizionari per convertire tra label testuali (es. "happy") e indici numerici (0..7) usati dal modello
LABEL2IDX = {lab: i for i, lab in enumerate(EMOTIONS)}
IDX2LABEL = {i: lab for lab, i in LABEL2IDX.items()}


def list_ravdess_files(data_root: str) -> List[str]:
    return glob(os.path.join(data_root, "Actor_*", "*.wav"))


def parse_ravdess_filename(filepath: str) -> Dict[str, str]:
    base = os.path.basename(filepath)
    stem = os.path.splitext(base)[0]
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Filename non valido RAVDESS: {base}")

    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6],
    }


def filter_audio_speech(filepaths: List[str]) -> List[str]:
    out = []
    for fp in filepaths:
        info = parse_ravdess_filename(fp)
        if info["modality"] == "03" and info["vocal_channel"] == "01":
            out.append(fp)
    return out


def extract_label_idx(filepath: str) -> int:
    info = parse_ravdess_filename(filepath)
    emotion_id = info["emotion"]
    lab = EMOTION_MAP[emotion_id]
    return LABEL2IDX[lab]


class RavdessDataset(Dataset):
    """
    X: [1, n_mels, T] log-mel (standardizzato)
    y: scalar 0..7
    """

    def __init__(
        self,
        filepaths: List[str],
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        max_duration: float = 4,
        use_db: bool = True,
        top_db: Optional[float] = 80.0,
    ):
        self.filepaths = filepaths
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)

        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
        )

        self.to_db = T.AmplitudeToDB(stype="power", top_db=top_db) if use_db else None

    def __len__(self):
        return len(self.filepaths)

    def _load_audio(self, path: str) -> torch.Tensor:
        # soundfile: ritorna np array shape [N] o [N, C]
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)  # [N, C]
        wav = torch.from_numpy(wav_np).transpose(0, 1)  # [C, N]

        # mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample se serve (torchaudio.functional.resample)
        if sr != self.sample_rate:
            wav = F.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        # pad / truncate a durata fissa
        n = wav.size(1)
        if n < self.max_samples:
            pad = self.max_samples - n
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[:, : self.max_samples]

        # normalize
        wav = wav / (wav.abs().max() + 1e-9)
        return wav

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        y = extract_label_idx(path)

        wav = self._load_audio(path)   # [1, N]
        spec = self.mel(wav)           # [1, n_mels, T]
        if self.to_db is not None:
            spec = self.to_db(spec)

        # standardizzazione per sample
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        return spec, torch.tensor(y, dtype=torch.long)
