from moviepy.editor import VideoFileClip
import os
import pandas as pd

def extract_audio(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = VideoFileClip(video_path)
    audio_path = os.path.join(output_folder, "audio.wav")
    video.audio.write_audiofile(audio_path)
    print(f"Extracted audio saved to {audio_path}")

# Load dataset
dataset = pd.read_csv("data.csv")

# Process each video in the dataset
for index, row in dataset.iterrows():
    video_path = row["video_path"].replace("\\", "/")  # Replace backslashes with forward slashes
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video name without extension
    output_folder = os.path.join("audio", video_name)  # Create a folder for each video's audio

    # Extract audio
    extract_audio(video_path, output_folder)