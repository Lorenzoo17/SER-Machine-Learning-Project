import torch
import torch.nn as nn


class CRNNBaseline(nn.Module):
    """
    CRNN conforme allo schema e alla tabella fornita.
    
    Input:  [B, 1, 128, 251]
    Output: [B, num_classes]
    """

    def __init__(self, num_classes=6):
        super().__init__()

        # =========================
        # Feature Learning Blocks
        # =========================
        self.cnn = nn.Sequential(
            # FLB1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d((2, 2)),      # 64×401 → 32×200
            nn.Dropout2d(0.2),

            # FLB2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d((2, 2)),      # 32×200 → 16×100
            nn.Dropout2d(0.2),

            # FLB3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d((2, 2)),      # 16×100 → 8×50
            nn.Dropout2d(0.3),

            # FLB4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(alpha=1.0),
            nn.AdaptiveMaxPool2d((1, 1))  # chiude sempre correttamente
        )


        # =========================
        # LSTM
        # =========================
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256, # da 256 a 128
            num_layers=1,
            batch_first=True
        )

        # =========================
        # Classifier
        # =========================
        self.fc = nn.Linear(256, num_classes) # da 256 a 128

    def forward(self, x):
        """
        x: [B, 1, 128, 251]
        """

        # CNN
        x = self.cnn(x)              # [B, 128, 1, 1]

        # Reshape
        x = x.squeeze(-1).squeeze(-1)  # [B, 128]
        x = x.unsqueeze(1)             # [B, 1, 128]

        # LSTM
        out, _ = self.lstm(x)          # [B, 1, 256]
        out = out[:, -1, :]            # many-to-one

        # Dense
        logits = self.fc(out)          # [B, num_classes]
        return logits
