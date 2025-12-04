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
│   ├── 01_preprocessing.ipynb   # test funzioni di preprocessing
│   └── 02_training.ipynb        # sperimentazioni sul training
│── requirements.txt
│── README.md

