import os
import torchaudio
from speechbrain.inference import EncoderClassifier

# Set the audio backend
torchaudio.set_audio_backend("soundfile")

# Load model
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-commonlanguage_ecapa",
        savedir="tmp/lang-id"
    )
except Exception as e:
    print("❌ Failed to load SpeechBrain model:", e)
    exit(1)

# Path to audio file
audio_file = os.path.join("tmp", "audio.wav")
if not os.path.exists(audio_file):
    print(f"❌ Audio file not found: {audio_file}")
    exit(1)

# Load and classify audio
try:
    signal, fs = torchaudio.load(audio_file)
    prediction = classifier.classify_batch(signal)

    # Extract language label and confidence
    label = prediction[3][0]
    score = float(prediction[1].max()) * 100

    print("\n--- ACCENT / LANGUAGE IDENTIFICATION ---")
    print(f"Detected: {label}")
    print(f"Confidence Score: {score:.2f}%")
except Exception as e:
    print("❌ Error during classification:", e)
