import torch
import torch.nn as nn


class CRNNBaseline(nn.Module):
    """
    CRNN conforme allo schema e alla tabella fornita.
    
    Input:  [B, 1, 128, 251]
    Output: [B, num_classes]
    """

    def __init__(self, num_classes):
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
<<<<<<< Updated upstream
            nn.ELU(alpha=1.0),
            nn.AdaptiveMaxPool2d((1, 1))  # chiude sempre correttamente
=======
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
>>>>>>> Stashed changes
        )


        # =========================
        # LSTM
        # =========================
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256, 
            num_layers=1,
            batch_first=True
        )

        # =========================
        # Classifier
        # =========================
        self.fc = nn.Linear(256, num_classes) # da 256 a 128

    def forward(self, x):
    # CNN: [B, 1, F, T] -> [B, 128, F', T']
        x = self.cnn(x)

        # reshape per LSTM: [B, 128, F', T'] -> [B, T', 128*F']
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T', C, F']
        x = x.view(B, T, C * F)                 # [B, T', C*F']  (qui dovrebbe essere 512)

        # LSTM
        out, _ = self.lstm(x)                   # [B, T', 256]
        out = out[:, -1, :]                     # many-to-one

        # Classifier
        logits = self.fc(out)                   # [B, num_classes]
        return logits
