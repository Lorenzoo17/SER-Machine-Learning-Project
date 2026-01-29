import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    Baseline SER: CNN (feature extractor) + BiLSTM (temporal) + classifier.
    Input: [B, 1, n_mels, T]
    """

    def __init__(self, n_classes: int = 8, n_mels: int = 64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # mels/2, T/2
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # mels/4, T/4
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # mels/8, T/4  (preserva un po' il tempo)
            nn.Dropout(0.2),
        )

        # dopo CNN: [B, C, M', T']
        # vogliamo LSTM su T': quindi trasformiamo in [B, T', C*M']
        # M' = n_mels / 8 (se n_mels=64 => 8)
        m_reduced = n_mels // 8
        lstm_in = 128 * m_reduced

        self.rnn = nn.LSTM(
            input_size=lstm_in,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: [B, 1, M, T]
        x = self.cnn(x)  # [B, C, M', T']
        b, c, m, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()      # [B, T', C, M']
        x = x.view(b, t, c * m)                     # [B, T', C*M']
        x, _ = self.rnn(x)                          # [B, T', 2H]

        # pooling temporale (mean)
        x = x.mean(dim=1)                           # [B, 2H]
        logits = self.classifier(x)                 # [B, n_classes]
        return logits
