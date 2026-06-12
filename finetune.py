import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import soundfile as sf

# ------------------------------------------------
# Paramètres
# ------------------------------------------------
REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"
MODEL_SAVE = "models/wav2vec2_finetuned"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 48000
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 3e-5
MAX_FILES = 200

# ------------------------------------------------
# Dataset
# ------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir, max_files=MAX_FILES):
        self.files = []
        self.labels = []

        real_files = [f for f in os.listdir(real_dir)
                      if f.endswith(".wav") or f.endswith(".flac")][:max_files]
        fake_files = [f for f in os.listdir(fake_dir)
                      if f.endswith(".wav") or f.endswith(".flac")][:max_files]

        for f in real_files:
            self.files.append(os.path.join(real_dir, f))
            self.labels.append(0)

        for f in fake_files:
            self.files.append(os.path.join(fake_dir, f))
            self.labels.append(1)

        print(f"Dataset : {len(real_files)} reels, {len(fake_files)} fakes")

    def load_audio(self, path):
        try:
            audio, sr = sf.read(path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if len(audio) < CHUNK_SAMPLES:
                audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
            else:
                audio = audio[:CHUNK_SAMPLES]
            return audio
        except Exception as e:
            print(f"Erreur lecture {path} : {e}")
            return np.zeros(CHUNK_SAMPLES, dtype=np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = self.load_audio(self.files[idx])
        label = self.labels[idx]
        return audio, label


# ------------------------------------------------
# Chargement du modèle Wav2Vec2
# ------------------------------------------------
print("Telechargement et chargement de Wav2Vec2...")
print("Premiere fois : peut prendre 5-10 minutes selon la connexion.\n")

MODEL_NAME = "facebook/wav2vec2-base"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Geler les couches de base — fine-tuner uniquement le classifieur
for name, param in model.named_parameters():
    if any(x in name for x in ["classifier", "projector", "wav2vec2.encoder.layers.11", "wav2vec2.encoder.layers.10", "wav2vec2.encoder.layers.9"]):
        param.requires_grad = True
    else:
        param.requires_grad = False

print("Couches gelees : encodeur wav2vec2")
print("Couches entrainables : classifieur final\n")

# ------------------------------------------------
# Préparation du dataset
# ------------------------------------------------
dataset = AudioDataset(REAL_DIR, FAKE_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Train : {train_size} fichiers")
print(f"Validation : {val_size} fichiers\n")


def collate_fn(batch):
    audios, labels = zip(*batch)
    inputs = feature_extractor(
        list(audios),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CHUNK_SAMPLES
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels


train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------
# Boucle d'entraînement
# ------------------------------------------------
print(f"Debut du fine-tuning pour {EPOCHS} epochs...\n")
best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1).numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Validation
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train — Loss: {total_loss/len(train_loader):.4f}  Acc: {train_acc*100:.1f}%  F1: {train_f1:.4f}")
    print(f"  Val   — Acc: {val_acc*100:.1f}%  F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        model.save_pretrained(MODEL_SAVE)
        feature_extractor.save_pretrained(MODEL_SAVE)
        print(f"  Meilleur modele sauvegarde (F1={val_f1:.4f})")

    print()

print(f"Fine-tuning termine. Meilleur F1 : {best_f1:.4f}")
print(f"Modele sauvegarde dans : {MODEL_SAVE}")