import torch
import torch.nn as nn


class CRNNBaseline(nn.Module):
    """
    Baseline CRNN per Speech Emotion Recognition.
    Input atteso:  X shape [B, 1, 64, 401]
    Output:        logits shape [B, 8]  (NO softmax: si usa CrossEntropyLoss)
    """

    def __init__(
        self,
        num_classes: int = 8,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # =========================
        # 1) CNN: estrazione feature locali (tempo/frequenza)
        # =========================
        # Nota: usiamo pool (2,2) per ridurre sia frequenza che tempo in modo controllato.
        # Input: [B, 1, 64, 401]

        # Qui facciamo batch normalization per rendere i layer più robusti ai cambiamenti dei pesi e per ridurre l'overfitting.

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # -> [B, 16, 64, 401]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),            # -> [B, 16, 32, 200]

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> [B, 32, 32, 200]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),            # -> [B, 32, 16, 100]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> [B, 64, 16, 100]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),            # -> [B, 64, 8, 50]
        )

        self.dropout = nn.Dropout(dropout)

        # ========================= 
        # 2) LSTM: modellazione temporale
        # =========================
        # Dopo la CNN avremo: [B, C, F, T] -> [Batch size, Channel (ora è in scala di grigio), Frequency, Time]
        # Convertiamo in sequenza sul tempo: [B, T, C*F]
        # Qui C=64, F=8 -> feature_dim = 512
        feature_dim = 64 * 8

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,  # guarda sia passato che futuro
        )

        # =========================
        # 3) Classificatore finale
        # =========================
        # Se bidirectional=True, la dimensione in uscita è 2*lstm_hidden
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward:
        x: [B, 1, 64, 401]
        return: [B, 8]
        """
        # CNN
        x = self.cnn(x)             # [B, 64, 8, 50] (con la pipeline attuale)
        x = self.dropout(x)

        # Reshape per LSTM: [B, C, F, T] -> [B, T, C*F]
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)   # [B, T, C, F]
        x = x.contiguous().view(B, T, C * F)  # [B, T, 512]

        # LSTM
        out, _ = self.lstm(x)       # out: [B, T, 2*lstm_hidden]
        out = self.dropout(out)

        # Aggregazione temporale:
        # prendiamo l'ultimo frame temporale (semplice e standard per baseline)
        out_last = out[:, -1, :]    # [B, 2*lstm_hidden],  out[:, -1, :] stato finale che ha visto tutta la sequenza

        # Andiamo a prendere l'ultimo frame temporale e lo classifichiamo. Usiamo LSTM in modalità many-to-one.

        # Classificazione finale
        logits = self.classifier(out_last)  # [B, 8]
        return logits
