# SER-Machine-Learning-Project
## STRUTTURA DELLE CARTELLE
project/
│── data/                        # dove tenere i file RAVDESS  
│── src/                         # tutto il codice “serio”  
│   ├── preprocessing/           # pulizia e trasformazione dei dati  
│   │   └── dataset.py           # Dataset PyTorch + mel-spectrogram + augmentation  
│   ├── models/                  # tutte le architetture  
│   │   └── crnn.py              # CNN+LSTM  
│   ├── training/                # training ed utilities  
│   │   └── train.py             # ciclo di addestramento  
│   └── evaluation/              # metriche e analisi finale  
│       └── metrics.py           # accuracy, F1, confusion matrix  
│── notebooks/                   # solo esplorazione e debug  
│   ├── preprocessing.ipynb   # test funzioni di preprocessing  
│   └── training.ipynb        # sperimentazioni sul training  
│── README.md

## TRACCIA DEL PROGGETTO
1. Dataset (RAVDESS + split speaker-independent)
2. Architettura (CNN + LSTM)
3. Training setup (preprocessing, augmentations, optimizer, loss…)
4. Valutazione (accuracy, precision, recall, F1, confusion matrix)

### Oggi 27/12/25 abbiamo fatto:
Dataset & preprocessing
- RAVDESS caricato da cartelle Actor_XX
- parsing del filename e labeling emozioni (ID → nome → indice)
- DataFrame di controllo con distribuzione emozioni/attori
- pipeline audio → mono → resample 16k → pad/crop 4s
- log-Mel (64 × 401) con MelSpectrogram + AmplitudeToDB
- RavdessDataset PyTorch che restituisce (X, y)
- split speaker-independent: train / val / test per attori
- DataLoader funzionante + funzione per plottare uno spettrogramma  
  
QUINDI La parte “DATASET + PREPROCESSING” è fatta.

## PROSSIMO PASSO: ARCHITETTURA (CNN + LSTM)
## POI:
### TRAINING SET UP
### AUGUMENTATIONS
### VALUTAZIONE
