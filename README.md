# SER-Machine-Learning-Project
## STRUTTURA DELLE CARTELLE
project/
│── data/                        # dove tenere i file RAVDESS  
│── src/                         # tutto il codice “serio”  
│   ├── 01preprocessing/           # pulizia e trasformazione dei dati  
│   │   └── dataset.py           # Dataset PyTorch + mel-spectrogram + augmentation  
│   ├── 02models/                  # tutte le architetture  
│   │   └── crnn.py              # CNN+LSTM  
│   ├── 03training/                # training ed utilities  
│   │   └── train.py             # ciclo di addestramento  
│   └── 04evaluation/              # metriche e analisi finale  
│       └── metrics.py           # accuracy, F1, confusion matrix  
│── notebooks/                   # solo esplorazione e debug  
│   ├── preprocessing.ipynb   # test funzioni di preprocessing  
│   └── training.ipynb        # sperimentazioni sul training  
│── README.md

## DA FARE
Algoritmo per classificare i .wav in base a questa decodifica:   
- Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
- Vocal channel (01 = speech, 02 = song).
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
- Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
- Repetition (01 = 1st repetition, 02 = 2nd repetition).
- Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
