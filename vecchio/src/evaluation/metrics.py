# src/evaluation/metrics.py

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, test_loader, device="cpu"):
    """
    Valuta un modello sul test set e restituisce:
    - accuracy
    - precision/recall/f1 (macro)
    - confusion matrix
    - report per classe
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_targets, all_preds)

    report = classification_report(all_targets, all_preds, zero_division=0)

    return acc, precision, recall, f1, cm, report
