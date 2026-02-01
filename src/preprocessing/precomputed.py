import os
import torch
from torch.utils.data import Dataset

class PrecomputedDataset(Dataset):
    def __init__(self, root_dir, selected_actors):
        """
        Carica i file .pt filtrando per gli attori selezionati nel fold.
        """
        self.files = []
        # Verifica se la cartella esiste per evitare errori silenti
        if not os.path.exists(root_dir):
            print(f"ATTENZIONE: La cartella {root_dir} non esiste.")
            return

        all_pt_files = [f for f in os.listdir(root_dir) if f.endswith('.pt')]
        
        for f in all_pt_files:
            try:
                # Il formato atteso è Actor_XX_...
                actor_in_file = f.split('_')[1] 
                if actor_in_file in selected_actors:
                    self.files.append(os.path.join(root_dir, f))
            except IndexError:
                continue
        
        if len(self.files) == 0:
            print(f"AVVISO: Nessun file trovato per gli attori: {selected_actors}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Caricamento con gestione memoria
        item = torch.load(self.files[idx], map_location="cpu")
        spec = item["spec"].detach()
        
        # Standardizzazione dinamica (fondamentale per CRNN)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        
        label = item["label"]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.long()

        return spec, label