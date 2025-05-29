import streamlit as st
import whisper
import torchaudio
import os
import subprocess
from yt_dlp import YoutubeDL
from speechbrain.inference.classifiers import EncoderClassifier
from huggingface_hub import login
from dotenv import load_dotenv
import requests

# Prevent Streamlit from crashing due to torch class loading
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Load Hugging Face token from .env or environment
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Streamlit UI setup
st.set_page_config(page_title="🎙️ Accent & Transcriber", layout="centered")
st.title("🎙️ Happy Transcribing")
st.markdown("Paste a YouTube or direct MP4 video link to detect the accent and transcribe speech.")
st.markdown("Training Data models for accurate accent detection , please be patient! 😅 suggestions? Email:ochiengderick12@gmail.com")
video_url = st.text_input("📹 Paste YouTube or Direct MP4 Video URL")

# Language code to flag mapping
flag_map = {
    "English": "🇬🇧", "en": "🇬🇧",
    "Chinese_Taiwan": "🇹🇼", "French": "🇫🇷", "Hindi": "🇮🇳",
    "Arabic": "🇸🇦", "Spanish": "🇪🇸",
    "Swahili": "🇰🇪", "Kiswahili": "🇰🇪"
}

def get_flag(lang_code):
    return flag_map.get(lang_code, "🌐")

# Only working, verified model
available_models = {
    "CommonLanguage ECAPA": "speechbrain/lang-id-commonlanguage_ecapa"
}
selected_model = st.selectbox("🧠 Choose Language ID Model", list(available_models.keys()))

# Download and convert video to audio
def download_audio(link, output_path="tmp/audio.wav"):
    os.makedirs("tmp", exist_ok=True)
    ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")

    if link.endswith(".mp4"):
        r = requests.get(link, stream=True)
        with open("tmp/input.mp4", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        subprocess.run([
            ffmpeg_path, "-y", "-i", "tmp/input.mp4",
            "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
            output_path
        ], check=True)
    else:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'tmp/audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])

# Main logic
if video_url:
    audio_path = "tmp/audio.wav"
    with st.spinner("🔄 Processing video..."):
        try:
            download_audio(video_url, audio_path)
            st.success("✅ Audio extracted successfully!")

            # Transcribe audio using Whisper
            st.subheader("📝 Transcription")
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(audio_path)
            transcription = result["text"]
            st.text_area("Transcript", transcription, height=180)

            # Language Identification
            st.subheader("🌍 Accent / Language Identification")
            try:
                signal, fs = torchaudio.load(audio_path)

                classifier = EncoderClassifier.from_hparams(
                    source=available_models[selected_model],
                    savedir="tmp/lang-id"
                )

                prediction = classifier.classify_batch(signal)
                lang = prediction[3][0]
                confidence = prediction[1][0].item()
                flag = get_flag(lang)

                st.markdown(f"**Detected:** {flag} `{lang}`")
                st.markdown(f"**Confidence Score:** `{confidence:.2f}`")

                if any(word in transcription.lower() for word in ["the", "and", "you", "your", "have", "is", "are"]) and "en" not in lang.lower():
                    st.warning("⚠️ Transcript appears to be English, but the model detected another language. Might be accent confusion or background noise.")

            except Exception as model_error:
                st.error(f"❌ Classification failed: {str(model_error)}")

        except subprocess.CalledProcessError as ffmpeg_error:
            st.error(f"🚨 ffmpeg error: {str(ffmpeg_error)}")

        except Exception as e:
            st.error(f"🚨 General error: {str(e)}")
