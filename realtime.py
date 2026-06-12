import torch
import numpy as np
import sounddevice as sd
import sys
import os
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# ------------------------------------------------
# Paramètres
# ------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 48000
SILENCE_THRESHOLD = 0.002  # Ajusté pour le signal brut non normalisé
DETECTION_THRESHOLD = 0.5
HISTORY_SIZE = 3

# ------------------------------------------------
# Chargement du modèle
# ------------------------------------------------
FINETUNED = "models/wav2vec2_finetuned"
PRETRAINED = "facebook/wav2vec2-base"

if os.path.exists(FINETUNED):
    MODEL_PATH = FINETUNED
    print("Modele fine-tune detecte.")
else:
    MODEL_PATH = PRETRAINED
    print("Modele de base utilise — fine-tuning recommande.")

print("Chargement du modele...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print("Modele charge avec succes.\n")

# ------------------------------------------------
# Afficher les périphériques
# ------------------------------------------------
print("=" * 60)
print("PERIPHERIQUES AUDIO DISPONIBLES :")
print("=" * 60)
print(sd.query_devices())
print("=" * 60)

default_device = sd.query_devices(kind='input')
print(f"\nMicrophone par defaut : {default_device['name']}")
print("Appuie sur Entree pour utiliser ce micro")
print("ou tape le numero d'un autre micro : ", end="")

choix = input().strip()
if choix == "":
    DEVICE_INDEX = None
    print(f"Utilisation du micro par defaut.")
else:
    DEVICE_INDEX = int(choix)
    info = sd.query_devices(DEVICE_INDEX)
    print(f"Microphone selectionne : {info['name']}")

# ------------------------------------------------
# Boucle principale
# ------------------------------------------------
print("\n" + "=" * 60)
print("   DETECTION DEEPFAKE ACTIVE — WAV2VEC2")
print(f"   Modele : {os.path.basename(MODEL_PATH)}")
print(f"   Seuil fake : {DETECTION_THRESHOLD}")
print("   Parle dans le micro — analyse toutes les 3 secondes")
print("   Ctrl+C pour arreter")
print("=" * 60)
print()

score_history = []
compteur = 1
fake_count = 0
real_count = 0
silence_count = 0

try:
    while True:
        print(f"[Segment {compteur}] Parle maintenant (3 sec)...", end=" ", flush=True)

        try:
            audio_np = sd.rec(
                frames=CHUNK_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                device=DEVICE_INDEX,
                blocking=True
            )
        except Exception as e:
            print(f"Erreur capture : {e}")
            compteur += 1
            continue

        audio_np = audio_np.flatten()

        # Calcul de l'énergie sur le signal REEL du micro (sans normalisation destructive)
        energie = np.sqrt(np.mean(audio_np ** 2))
        
        if energie < SILENCE_THRESHOLD:
            print(f"silence (energie={energie:.4f})")
            print()
            silence_count += 1
            compteur += 1
            continue

        # Préparation input — Le Feature Extractor s'occupe de la bonne normalisation
        inputs = feature_extractor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CHUNK_SAMPLES
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            fake_score = probs[0][1].item()

        score_history.append(fake_score)
        if len(score_history) > HISTORY_SIZE:
            score_history.pop(0)
        avg_fake_score = float(np.mean(score_history))
        avg_real_score = 1.0 - avg_fake_score

        if avg_fake_score >= DETECTION_THRESHOLD:
            verdict = "DEEPFAKE DETECTE"
            statut = "!!! DANGER !!!"
            fake_count += 1
        else:
            verdict = "AUDIO REEL"
            statut = "OK"
            real_count += 1

        barre_fake = "#" * int(avg_fake_score * 20)
        barre_real = "#" * int(avg_real_score * 20)

        print(f"{statut} — {verdict}")
        print(f"   Score brut     : fake={fake_score*100:.1f}%  reel={(1-fake_score)*100:.1f}%")
        print(f"   Moyenne ({HISTORY_SIZE} seg) : fake={avg_fake_score*100:.1f}%  reel={avg_real_score*100:.1f}%")
        print(f"   Reel : {avg_real_score*100:.1f}%  [{barre_real:<20}]")
        print(f"   Fake : {avg_fake_score*100:.1f}%  [{barre_fake:<20}]")
        print(f"   Energie voix   : {energie:.4f}")
        print()

        analyses = fake_count + real_count
        if analyses > 0 and analyses % 10 == 0:
            print("-" * 60)
            print(f"   STATISTIQUES ({analyses} segments analyses)")
            print(f"   Reels    : {real_count} ({real_count/analyses*100:.0f}%)")
            print(f"   Fakes    : {fake_count} ({fake_count/analyses*100:.0f}%)")
            print(f"   Silences : {silence_count}")
            print("-" * 60)
            print()

        compteur += 1

except KeyboardInterrupt:
    print("\nArret de la detection.")
    analyses = fake_count + real_count
    if analyses > 0:
        print("\n" + "=" * 60)
        print("   RAPPORT FINAL")
        print("=" * 60)
        print(f"   Total analyses  : {analyses}")
        print(f"   Reels           : {real_count} ({real_count/analyses*100:.0f}%)")
        print(f"   Fakes           : {fake_count} ({fake_count/analyses*100:.0f}%)")
        print(f"   Silences        : {silence_count}")
        if fake_count > 0:
            print(f"\n   ATTENTION : {fake_count} segments suspects detectes.")
        else:
            print(f"\n   Aucun deepfake detecte.")
        print("=" * 60)

print("Fermeture. Au revoir.")