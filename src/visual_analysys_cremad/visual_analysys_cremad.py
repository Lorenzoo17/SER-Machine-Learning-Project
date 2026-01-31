import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src.preprocessing.dataset import CREMA_IDX2LABEL

# =========================================================
# VISUAL ANALYSIS UTILS: Embeddings + t-SNE + Grad-CAM
# =========================================================

@torch.no_grad()
def extract_crnn_embeddings(model, loader, device):
    """
    Estrae embeddings dal CRNN senza modificare la classe:
    CNN -> BiLSTM -> mean pooling temporale (prima del classifier).
    """
    model.eval()
    embs, ys, preds = [], [], []

    for x, y in loader:
        x = x.to(device)

        feat = model.cnn(x)  # [B, C, M', T']
        b, c, m, t = feat.shape
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, T', C, M']
        feat = feat.view(b, t, c * m)                 # [B, T', C*M']

        seq, _ = model.rnn(feat)                      # [B, T', 2H]
        emb = seq.mean(dim=1)                         # [B, 2H]

        logits = model.classifier(emb)                # [B, n_classes]
        y_hat = logits.argmax(dim=1)

        embs.append(emb.detach().cpu())
        ys.append(y.detach().cpu())
        preds.append(y_hat.detach().cpu())

    return (
        torch.cat(embs, dim=0).numpy(),
        torch.cat(ys, dim=0).numpy(),
        torch.cat(preds, dim=0).numpy(),
    )


def tsne_project(embeddings, pca_dim=50, perplexity=30, seed=42):
    X = embeddings
    if pca_dim is not None and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed
    )
    return tsne.fit_transform(X)


def plot_fold_summary(Z, y_true, y_pred, fold_idx, test_speakers):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) t-SNE colored by true label
    ax = axes[0]
    for k in np.unique(y_true):
        idx = (y_true == k)
        ax.scatter(Z[idx, 0], Z[idx, 1], s=18, alpha=0.7, label=CREMA_IDX2LABEL[int(k)])
    ax.set_title("t-SNE (true labels)")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(fontsize=8, markerscale=1.1)

    # 2) Correct vs error
    ax = axes[1]
    correct = (y_true == y_pred)
    ax.scatter(Z[correct, 0], Z[correct, 1], s=15, alpha=0.5, label="correct")
    ax.scatter(Z[~correct, 0], Z[~correct, 1], s=40, alpha=0.9, marker="x", label="error")
    ax.set_title("t-SNE (errors)")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend()

    # 3) Confusion matrix
    ax = axes[2]
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")

    ticks = np.arange(6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([CREMA_IDX2LABEL[i] for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([CREMA_IDX2LABEL[i] for i in ticks])

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Fold {fold_idx + 1} | Test speakers: {test_speakers}", fontsize=14)

    plt.tight_layout()
    plt.show()


def find_last_conv2d(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Nessun nn.Conv2d trovato: non posso fare Grad-CAM.")
    return last


class GradCAM:
    """
    Grad-CAM generico sul last Conv2D.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_a = target_layer.register_forward_hook(self._forward_hook)
        self.hook_g = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.hook_a.remove()
        self.hook_g.remove()

    def __call__(self, x, class_idx=None):
        was_training = self.model.training

        # per backward stabile su LSTM/cuDNN
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        A = self.activations
        G = self.gradients
        if A is None or G is None:
            raise RuntimeError("Hooks Grad-CAM non hanno catturato activations/gradients.")

        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)

        cam_up = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        if not was_training:
            self.model.eval()

        return cam_up.detach().cpu().squeeze(0).squeeze(0), class_idx, logits.detach().cpu()


def gradcam_summary(model, loader, device, n_examples=4, seed=42):
    rng = np.random.RandomState(seed)
    n_examples = min(n_examples, len(loader.dataset))
    idxs = rng.choice(len(loader.dataset), size=n_examples, replace=False)

    target_layer = find_last_conv2d(model)
    cam_engine = GradCAM(model, target_layer)

    fig, axes = plt.subplots(len(idxs), 1, figsize=(12, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = [axes]

    model.eval()
    for ax, i in zip(axes, idxs):
        spec, y = loader.dataset[i]        # spec: [1, n_mels, T]
        x = spec.unsqueeze(0).to(device)   # [1,1,n_mels,T]
        y = int(y.item())

        cam, _, logits = cam_engine(x, class_idx=None)
        pred = int(logits.argmax(dim=1).item())

        spec_np = spec.squeeze(0).cpu().numpy()
        cam_np = cam.numpy()

        ax.imshow(spec_np, aspect="auto", origin="lower")
        ax.imshow(cam_np, aspect="auto", origin="lower", alpha=0.45)
        ax.set_title(f"true={CREMA_IDX2LABEL[y]} | pred={CREMA_IDX2LABEL[pred]}", fontsize=11)
        ax.set_xlabel("time")
        ax.set_ylabel("mel bins")

    plt.tight_layout()
    plt.show()
    cam_engine.close()