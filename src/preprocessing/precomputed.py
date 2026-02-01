import os
import torch
from torch.utils.data import Dataset

class PrecomputedDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pt")]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Carichiamo il file
        item = torch.load(self.files[idx], map_location="cpu")
        
        # .detach() rimuove ogni traccia di calcolo precedente (gradiente)
        spec = item["spec"].detach() 
        label = torch.tensor(item["label"], dtype=torch.long)
        
        return spec, label