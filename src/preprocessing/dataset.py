import os
from glob import glob
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

import soundfile as sf
import random
import numpy as np



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
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_duration: float = 6,
        use_db: bool = True,
        top_db: Optional[float] = 80.0,
        augmentation: bool = False,
        aug_config: Optional[dict] = None,
    ):
        self.filepaths = filepaths
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.augmentation = augmentation
        self.aug_config = aug_config or {}

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
        try:
            wav, sr = torchaudio.load(path)  # [C, N]
        except Exception as e:
            print("FAILED torchaudio:", path, "|", e)
            wav = torch.zeros(1, self.max_samples)
            sr = self.sample_rate

        # mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample
        if sr != self.sample_rate:
            wav = F.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        # pad / truncate
        n = wav.size(1)
        if n < self.max_samples:
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - n))
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

    def apply_augmentation(self, wav: torch.Tensor) -> torch.Tensor:
        cfg = self.aug_config

        # Applica augmentation solo al 70% dei campioni
        if random.random() > 0.7:
            return wav
    
        # 1) Random Gain
        if cfg.get("gain", False):
            gmin, gmax = cfg.get("gain_db", (-6, 6))
            gain_db = random.uniform(gmin, gmax)
            wav = wav * (10 ** (gain_db / 20))

        # 2) Time Shift (in secondi)
        if cfg.get("time_shift", False):
            max_s = cfg.get("time_shift_s", 0.10)  # 100 ms default
            max_shift = int(max_s * self.sample_rate)
            shift = random.randint(-max_shift, max_shift)
            wav = torch.roll(wav, shifts=shift, dims=1)

        # 3) Additive Noise (gaussiano, SNR approx)
        if cfg.get("noise", False):
            snr_min, snr_max = cfg.get("snr_db", (10, 30))
            snr_db = random.uniform(snr_min, snr_max)

            signal_power = wav.pow(2).mean()
            noise = torch.randn_like(wav)
            noise_power = noise.pow(2).mean()

            # scala noise per ottenere SNR desiderato
            snr_linear = 10 ** (snr_db / 10)
            scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-9))
            wav = wav + scale * noise

        # 4) Simple Reverb (IR sintetica: decadimento esponenziale)
        if cfg.get("reverb", False) and random.random() < 0.2:
            ir_len = int(cfg.get("reverb_ir_s", 0.12) * self.sample_rate)  # 120ms
            decay = cfg.get("reverb_decay", 0.3)
            t = torch.arange(ir_len, device=wav.device).float()
            ir = torch.exp(-t / (decay * self.sample_rate)).unsqueeze(0)  # [1, L]
            ir = ir / (ir.sum() + 1e-9)

            # conv1d: input [B=1,C=1,N] -> qui wav è [1,N], quindi aggiungi batch
            wav_b = wav.unsqueeze(0)  # [1,1,N]
            ir_b = ir.unsqueeze(0)    # [1,1,L]
            wav = torch.nn.functional.conv1d(wav_b, ir_b, padding=ir_len//2).squeeze(0)

        # clamp finale per sicurezza
        wav = torch.clamp(wav, -1.0, 1.0)
        return wav








# PROVA NUOVO DATA SET CREMA D

# CREMA-D emotions (6 classi standard)
CREMA_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

CREMA_LABEL2IDX = {lab: i for i, lab in enumerate(CREMA_EMOTIONS)}
CREMA_IDX2LABEL = {i: lab for lab, i in CREMA_LABEL2IDX.items()}

def list_cremad_files(data_root: str):
    return glob(os.path.join(data_root, "AudioWAV", "*.wav"))

def parse_cremad_filename(filepath: str):
    base = os.path.basename(filepath)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")

    speaker = parts[0]           
    emotion_code = parts[2]     

    code_map = {
        "NEU": "neutral",
        "HAP": "happy",
        "SAD": "sad",
        "ANG": "angry",
        "FEA": "fearful",
        "DIS": "disgust",
    }

    return {
        "speaker": speaker,
        "emotion": code_map[emotion_code],
    }

def extract_cremad_label_idx(filepath: str):
    info = parse_cremad_filename(filepath)
    return CREMA_LABEL2IDX[info["emotion"]]

class CremadDataset(RavdessDataset):
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        y = extract_cremad_label_idx(path)

        wav = self._load_audio(path)
        spec = self.mel(wav)
        if self.to_db is not None:
            spec = self.to_db(spec)

        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec, torch.tensor(y, dtype=torch.long)
