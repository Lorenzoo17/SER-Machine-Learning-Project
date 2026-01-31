import random
import numpy as np
import torch

# Imposta i seed (random/numpy/torch) per rendere gli esperimenti riproducibili
# In modo che tutti i parametri random siano gli stessi ad ogni training (facilita confronti)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinismo (può rallentare un po')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, loader, device, use_dann: bool = False):
    """
    Compatibile con DataLoader che produce:
      - (x, y)
      - (x, y, d)   (d = domain/speaker id)

    Se use_dann=True, assume che model(x, grl_lambda=0.0) ritorni:
      emo_logits, dom_logits, z
    Se use_dann=False, assume che model(x) ritorni:
      emo_logits
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        if use_dann:
            emo_logits, _, _ = model(x, grl_lambda=0.0)
        else:
            emo_logits = model(x)

        loss = F.cross_entropy(emo_logits, y)

        total_loss += loss.item() * y.size(0)
        correct += (emo_logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total
