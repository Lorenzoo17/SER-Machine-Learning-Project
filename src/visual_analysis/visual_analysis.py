
# VISUAL ANALYSIS
# t-SNE on embeddings + Grad-CAM su spettrogrammi

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.preprocessing.dataset import IDX2LABEL, parse_ravdess_filename

#t-SNE on embeddings 
@torch.no_grad()
def extract_crnn_embeddings(model, loader, device, return_paths=True):
    """
    Estrae embeddings dal CRNN SENZA modificare la classe.
    Qui definiamo "embedding" come il vettore dopo:
    CNN -> BiLSTM -> mean pooling temporale, prima del classifier.
    
    Ritorna:
      - emb: [N, D] numpy
      - y_true: [N] numpy
      - y_pred: [N] numpy
      - paths: lista file (opzionale)
    """
    model.eval()
    embs = []
    ys = []
    preds = []
    paths = []

    # Se il tuo Dataset non restituisce il path, lo ricaviamo via loader.dataset.filepaths
    # assumendo che loader NON shuffli (true per val/test nel tuo notebook)
    dataset_paths = getattr(loader.dataset, "filepaths", None)
    global_index = 0

    for batch in loader:
        x, y = batch
        x = x.to(device)

        # --- ricostruzione "forward fino all'embedding" basata sulla tua CRNN ---
        feat = model.cnn(x)                      # [B, C, M', T']
        b, c, m, t = feat.shape
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, T', C, M']
        feat = feat.view(b, t, c * m)                 # [B, T', C*M']
        seq, _ = model.rnn(feat)                      # [B, T', 2H]
        emb = seq.mean(dim=1)                         # [B, 2H]  <-- embedding

        logits = model.classifier(emb)                # [B, n_classes]
        y_hat = logits.argmax(dim=1)

        embs.append(emb.detach().cpu())
        ys.append(y.detach().cpu())
        preds.append(y_hat.detach().cpu())

        if return_paths and dataset_paths is not None:
            bs = y.size(0)
            paths.extend(dataset_paths[global_index: global_index + bs])
            global_index += bs

    embs = torch.cat(embs, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()

    return embs, ys, preds, paths


def tsne_project(embeddings, pca_dim=50, tsne_perplexity=30, tsne_lr="auto", seed=42):
    """
    PCA (opzionale) + t-SNE -> 2D
    """
    X = embeddings
    if pca_dim is not None and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate=tsne_lr,
        init="pca",
        random_state=seed
    )
    Z = tsne.fit_transform(X)
    return Z


def plot_tsne_by_label(Z, y_true, title="t-SNE (colored by true label)"):
    plt.figure(figsize=(10, 7))
    for k in np.unique(y_true):
        idx = (y_true == k)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=18, alpha=0.75, label=IDX2LABEL[int(k)])
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend(markerscale=1.2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_tsne_errors_and_speakers(Z, y_true, y_pred, paths, title="t-SNE (errors + speaker id)"):
    """
    Plot originale: evidenziamo errori e aggiungiamo speaker id.
    - corretto: marker 'o'
    - errato: marker 'x'
    Colore: emotion vera
    """
    correct = (y_true == y_pred)

    # Speaker/actor id dal filename (RAVDESS Actor_XX)
    speakers = []
    for fp in paths:
        actor = parse_ravdess_filename(fp)["actor"]
        speakers.append(actor)
    speakers = np.array(speakers)

    plt.figure(figsize=(10, 7))

    # Per non avere 24 legende, mettiamo speaker id come testo solo per alcuni punti (campionamento)
    # e ci concentriamo su errori vs corretti
    for k in np.unique(y_true):
        idx_k = (y_true == k)

        # corretti
        idx_ok = idx_k & correct
        plt.scatter(Z[idx_ok, 0], Z[idx_ok, 1], s=16, alpha=0.65, marker="o")

        # errati
        idx_bad = idx_k & (~correct)
        if idx_bad.any():
            plt.scatter(Z[idx_bad, 0], Z[idx_bad, 1], s=40, alpha=0.9, marker="x")

    # Aggiungi qualche speaker label sui punti sbagliati (molto utile per capire leakage)
    bad_idx = np.where(~correct)[0]
    for i in bad_idx[:30]:  # limita per non “sporcare” il grafico
        plt.text(Z[i, 0], Z[i, 1], speakers[i], fontsize=8, alpha=0.85)

    plt.title(title + "  (x = misclassified; text = speaker id on some errors)")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    plt.show()

#Grad-CAM on spectrograms
def find_last_conv2d(module: nn.Module):
    """
    Trova automaticamente l'ultimo nn.Conv2d dentro un modello (o sotto-moduli).
    Così se cambi architettura, spesso non devi cambiare nulla.
    """
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Nessun nn.Conv2d trovato: non posso fare Grad-CAM.")
    return last


class GradCAM:
    """
    Grad-CAM generico:
    - hook su ultimo Conv2d (o quello che passi tu)
    - produce heatmap (H x W) allineata allo spettrogramma input
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_a = target_layer.register_forward_hook(self._forward_hook)
        self.hook_g = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out  # [B, C, H, W]

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # [B, C, H, W]

    def close(self):
        self.hook_a.remove()
        self.hook_g.remove()

    def __call__(self, x, class_idx=None):
        import torch.nn as nn
        import torch.nn.functional as F

        # salva stato originale
        was_training = self.model.training

        # serve per far funzionare backward su cuDNN LSTM
        self.model.train()

        # disattiva dropout per mappe stabili
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)  # [1, n_classes]
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        # attivazioni e gradienti dal layer target
        A = self.activations          # [1, C, H, W]
        G = self.gradients            # [1, C, H, W]
        if A is None or G is None:
            raise RuntimeError("Hooks Grad-CAM non hanno catturato activations/gradients. Controlla target_layer.")

        # pesi: global average pooling dei gradienti
        weights = G.mean(dim=(2, 3), keepdim=True)       # [1, C, 1, 1]

        # somma pesata delle attivazioni
        cam = (weights * A).sum(dim=1, keepdim=True)     # [1, 1, H, W]
        cam = F.relu(cam)

        # normalizzazione
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)

        # upsample alla size dello spettrogramma di input
        cam_up = F.interpolate(
            cam,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False
        )  # [1, 1, n_mels, T]

        # ripristina stato originale
        if not was_training:
            self.model.eval()

        return cam_up.detach().cpu().squeeze(0).squeeze(0), class_idx, logits.detach().cpu()



def show_gradcam_on_spectrogram(spec, cam, title="", true_label=None, pred_label=None):
    """
    spec: [1, n_mels, T] tensor (cpu o gpu)
    cam:  [n_mels, T] tensor cpu (da GradCAM)
    """
    spec = spec.detach().cpu().squeeze(0)  # [n_mels, T]

    plt.figure(figsize=(12, 4))
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.imshow(cam, aspect="auto", origin="lower", alpha=0.45)  # overlay
    t = title
    if true_label is not None and pred_label is not None:
        t += f" | true={true_label} pred={pred_label}"
    plt.title(t)
    plt.xlabel("time")
    plt.ylabel("mel bins")
    plt.tight_layout()
    plt.show()


def gradcam_demo(model, loader, device, n_examples=10, seed=42, class_mode="pred"):
    """
    Seleziona n_examples campioni casuali e mostra Grad-CAM.
    class_mode:
      - "pred": spiega la classe predetta
      - "true": spiega la classe vera
    """
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(loader.dataset), size=min(n_examples, len(loader.dataset)), replace=False)

    target_layer = find_last_conv2d(model)
    cam_engine = GradCAM(model, target_layer)

    model.eval()
    for i in idxs:
        spec, y = loader.dataset[i]  # spec: [1, n_mels, T]
        x = spec.unsqueeze(0).to(device)  # [1,1,n_mels,T]
        y = int(y.item())

        if class_mode == "true":
            cam, cidx, logits = cam_engine(x, class_idx=y)
        else:
            cam, cidx, logits = cam_engine(x, class_idx=None)

        pred = int(logits.argmax(dim=1).item())
        show_gradcam_on_spectrogram(
            spec,
            cam,
            title="C) Grad-CAM on log-mel spectrogram",
            true_label=IDX2LABEL[y],
            pred_label=IDX2LABEL[pred]
        )
    cam_engine.close()