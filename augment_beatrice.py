import numpy as np
import soundfile as sf
import librosa
import os
from scipy.signal import resample

SOURCE_FILE = "beatrice_source.wav"
OUTPUT_DIR = "dataset/fake"
TARGET_SR = 16000
NUM_GENERATE = 100  # nombre de fichiers à générer

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger le fichier source
print("Chargement du fichier source...")
audio, sr = sf.read(SOURCE_FILE)

# Convertir en mono
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Rééchantillonner à 16000 Hz
if sr != TARGET_SR:
    num_samples = int(len(audio) * TARGET_SR / sr)
    audio = resample(audio, num_samples)
    sr = TARGET_SR

audio = audio.astype(np.float32)
CHUNK = TARGET_SR * 3  # 3 secondes par fichier

print(f"Génération de {NUM_GENERATE} fichiers...\n")

count = 0
for i in range(NUM_GENERATE):
    # Prendre un segment aléatoire de 3 secondes
    if len(audio) > CHUNK:
        start = np.random.randint(0, len(audio) - CHUNK)
        segment = audio[start:start + CHUNK].copy()
    else:
        segment = np.pad(audio, (0, CHUNK - len(audio)))

    # Appliquer une augmentation aléatoire
    aug_type = i % 5

    if aug_type == 0:
        # Changement de vitesse léger
        rate = np.random.uniform(0.85, 1.15)
        segment = librosa.effects.time_stretch(segment, rate=rate)

    elif aug_type == 1:
        # Changement de pitch
        steps = np.random.uniform(-3, 3)
        segment = librosa.effects.pitch_shift(segment, sr=TARGET_SR, n_steps=steps)

    elif aug_type == 2:
        # Ajout de bruit léger
        noise = np.random.normal(0, 0.003, len(segment))
        segment = segment + noise

    elif aug_type == 3:
        # Changement de volume
        factor = np.random.uniform(0.6, 1.4)
        segment = segment * factor

    elif aug_type == 4:
        # Combinaison pitch + bruit
        steps = np.random.uniform(-2, 2)
        segment = librosa.effects.pitch_shift(segment, sr=TARGET_SR, n_steps=steps)
        noise = np.random.normal(0, 0.002, len(segment))
        segment = segment + noise

    # Normaliser et ajuster la taille
    if np.max(np.abs(segment)) > 0:
        segment = segment / np.max(np.abs(segment)) * 0.9

    if len(segment) < CHUNK:
        segment = np.pad(segment, (0, CHUNK - len(segment)))
    else:
        segment = segment[:CHUNK]

    segment = segment.astype(np.float32)

    # Sauvegarder
    out_path = os.path.join(OUTPUT_DIR, f"beatrice_{i:04d}.wav")
    sf.write(out_path, segment, TARGET_SR)
    count += 1

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{NUM_GENERATE} fichiers générés...")

print(f"\nTerminé ! {count} fichiers Beatrice ajoutés dans {OUTPUT_DIR}")