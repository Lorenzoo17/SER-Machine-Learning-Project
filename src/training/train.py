import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Calcola accuracy batch (0..1) a partire dai logits."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y).sum().item()
    return correct / y.size(0)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Esegue 1 epoch.
    - Se optimizer è None -> evaluation (no backward)
    - Se optimizer è dato -> training (backward + step)
    Ritorna: (loss_media, accuracy_media)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    # In evaluation non calcoliamo gradienti
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(X)               # [B, 8]
            loss = criterion(logits, y)     # CrossEntropyLoss

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(logits, y) * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    save_name: str = "best_model.pt",
) -> Dict[str, list]:
    """
    Training completo con validazione e salvataggio best model.
    Ritorna uno storico (loss/acc train e val).
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,   # None => eval
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Salviamo il modello migliore sulla validation (loss minima)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )
            print(f"  -> saved best model to: {save_path}")

    return history
