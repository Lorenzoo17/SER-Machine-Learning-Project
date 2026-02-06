# src/experiment_config.py

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class ExperimentConfig:
    # --- toggles ---
    speaker_independent: bool = True
    augmentation: bool = True

    # --- CV / split ---
    n_folds: int = 6
    seed: int = 42

    # --- training ---
    batch_size: int = 64
    epochs: int = 70
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # --- audio / features ---
    sample_rate: int = 16000
    n_mels: int = 64
    max_duration: float = 4.0

    # --- augmentation  ---
    aug_cfg: Optional[Dict] = None

    def build_aug_cfg(self) -> Dict:
        
        if self.aug_cfg is not None:
            return self.aug_cfg

        return {
            "gain": True,
            "gain_db": (-3, 3),

            "time_shift": True,
            "time_shift_s": 0.03,

            "noise": True,
            "snr_db": (25, 40),
        }
