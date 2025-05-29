import os
import torchaudio
torchaudio.set_audio_backend("soundfile")  # <- This sets the backend explicitly

from speechbrain.inference import EncoderClassifier  # Updated per deprecation warning


# Load pretrained language/accent identification model
classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa")


# Path to audio
audio_file = os.path.join("tmp", "audio.wav")


# Run classification
signal, fs = torchaudio.load(audio_file)
prediction = classifier.classify_batch(signal)

# Extract label and score
label = prediction[3][0]
score = float(prediction[1].max()) * 100

print("\n--- ACCENT / LANGUAGE IDENTIFICATION ---")
print(f"Detected: {label}")
print(f"Confidence Score: {score:.2f}%")
