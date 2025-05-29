import whisper
import os

def transcribe_audio(audio_path):
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # You can use "tiny", "base", "small", "medium", "large"
    
    print("Transcribing...")
    result = model.transcribe(audio_path)
    
    print("\n--- TRANSCRIPTION RESULT ---\n")
    print(result["text"])
    return result["text"]

if __name__ == "__main__":
    audio_file = os.path.join("tmp", "audio.wav")
    
    if os.path.exists(audio_file):
        transcribe_audio(audio_file)
    else:
        print("Audio file not found. Please run extract_audio.py first.")
