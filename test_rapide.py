import torch
import numpy as np
import soundfile as sf
import os
import sys
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.metrics import f1_score, accuracy_score

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 48000
REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"
MAX_TEST = 20

print("Chargement du modele FINETUNÉ local...")
MODEL_PATH = "models/wav2vec2_finetuned"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def load_audio(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if len(audio) < CHUNK_SAMPLES:
        audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
    else:
        audio = audio[:CHUNK_SAMPLES]
    return audio

preds = []
labels = []

real_files = [f for f in os.listdir(REAL_DIR) if f.endswith(".wav")][:MAX_TEST]
fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(".wav")][:MAX_TEST]

print(f"Test sur {len(real_files)} reels et {len(fake_files)} fakes...\n")

for f in real_files:
    audio = load_audio(os.path.join(REAL_DIR, f))
    inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
        prob = torch.softmax(out.logits, dim=-1)
        pred = torch.argmax(prob).item()
    preds.append(pred)
    labels.append(0)
    print(f"REEL  : {f[:30]} → {'REEL' if pred==0 else 'FAKE'} (fake={prob[0][1].item()*100:.1f}%)")

for f in fake_files:
    audio = load_audio(os.path.join(FAKE_DIR, f))
    inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
        prob = torch.softmax(out.logits, dim=-1)
        pred = torch.argmax(prob).item()
    preds.append(pred)
    labels.append(1)
    print(f"FAKE  : {f[:30]} → {'REEL' if pred==0 else 'FAKE'} (fake={prob[0][1].item()*100:.1f}%)")

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, zero_division=0)

print(f"\nSans entrainement — Accuracy: {acc*100:.1f}%  F1: {f1:.4f}")
print("\nSi F1 > 0.6 avant entrainement : le fine-tuning va bien marcher.")
print("Si F1 < 0.5 : le dataset n'est pas adapte, il faut changer de donnees.")