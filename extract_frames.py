import cv2
import os
import pandas as pd

def extract_frames(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count // frame_interval} frames from {video_path}")

# Load dataset
dataset = pd.read_csv("data.csv")

# Process each video in the dataset
for index, row in dataset.iterrows():
    video_path = row["video_path"].replace("\\", "/")  # Replace backslashes with forward slashes
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video name without extension
    output_folder = os.path.join("frames", video_name)  # Create a folder for each video's frames

    # Extract frames
    extract_frames(video_path, output_folder)