import os
import whisper
from yt_dlp import YoutubeDL
import imageio_ffmpeg

TMP_DIR = "tmp"
AUDIO_FILENAME = "audio.wav"
AUDIO_PATH = os.path.join(TMP_DIR, AUDIO_FILENAME)


def ensure_tmp_dir():
    """Ensure the tmp directory exists."""
    os.makedirs(TMP_DIR, exist_ok=True)


def download_audio(video_url):
    """Download and convert YouTube audio to .wav format."""
    ensure_tmp_dir()
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    options = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(TMP_DIR, 'audio.%(ext)s'),
        'ffmpeg_location': ffmpeg_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False
    }

    try:
        print("🔄 Downloading audio...")
        with YoutubeDL(options) as ydl:
            ydl.download([video_url])
        print(f"✅ Audio saved as: {AUDIO_PATH}")
    except Exception as e:
        print(f"❌ Failed to download audio: {e}")
        exit(1)


def transcribe_audio():
    """Transcribe the audio file using Whisper."""
    if not os.path.exists(AUDIO_PATH):
        print("❌ Audio file not found. Make sure to download it first.")
        exit(1)

    print("🧠 Loading Whisper model (base)...")
    model = whisper.load_model("base")

    print("📝 Transcribing audio...")
    result = model.transcribe(AUDIO_PATH)
    print("\n--- TRANSCRIPTION RESULT ---\n")
    print(result["text"])
    return result["text"]


if __name__ == "__main__":
    video_url = input("🎥 Paste YouTube video URL: ").strip()

    if not video_url:
        print("❌ No URL provided. Exiting.")
        exit(1)

    download_audio(video_url)
    transcribe_audio()
