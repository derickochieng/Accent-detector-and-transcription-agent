import os
from yt_dlp import YoutubeDL

def download_audio_from_url(video_url, output_dir="tmp"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    options = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, 'audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False
    }

    with YoutubeDL(options) as ydl:
        print("Downloading and extracting audio...")
        ydl.download([video_url])
        print("Audio saved to:", os.path.join(output_dir, "audio.wav"))

# Example usage
if __name__ == "__main__":
    url = input("Paste video URL: ")
    download_audio_from_url(url)
