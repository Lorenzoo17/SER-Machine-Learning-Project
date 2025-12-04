import os

# Mappa emozioni RAVDESS
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_emotion_label(filepath: str) -> str:
    """
    Estrae il label (emozione) dal nome del file RAVDESS.
    Esempio: 03-01-08-02-02-01-12.wav → 'surprised'
    """
    filename = os.path.basename(filepath)
    parts = filename.split('-')
    emotion_id = parts[2]  # terzo numero → emozione
    return EMOTION_MAP[emotion_id]
