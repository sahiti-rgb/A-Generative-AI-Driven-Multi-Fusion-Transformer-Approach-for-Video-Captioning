import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import librosa
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from collections import Counter
import glob


# Load pre-trained BLIP model captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_audio_mfcc(audio_path, sr=16000, n_mfcc=40):
    """
    Returns a 1D numpy vector of shape (n_mfcc,) = mean MFCC over time.
    """
    y, _sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # shape: (n_mfcc,)

def list_frames_from_folder(frames_folder, extensions=("jpg","jpeg","png")):
    """
    Returns sorted list of image file paths from a frames folder.
    """
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(frames_folder, f"*.{ext}"))
    files = sorted(files)
    return files

def generate_image_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def analyze_audio(audio_path):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=None)
    # Extract features 
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def generate_video_caption(frames_folder, audio_path=None, device=None, max_length=30):
    """
    Bootstrapped audio-visual fusion:
    - Extract frames
    - For each frame: get BLIP vision features + generate frame caption
    - If audio_path given: compute MFCC mean, project to vision-dim, compute cosine similarity
      between audio-embedding and each frame-embedding; weight captions by similarity and pick top.
    - Else: fallback to majority-vote across frame captions (old behaviour).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    global model, processor
    model.to(device)
    model.eval()

    # 1) gather frames
    frame_files = list_frames_from_folder(frames_folder)
    if len(frame_files) == 0:
        return ""

    captions = []
    frame_feats = []  # list of tensors [1, hidden_dim]

    with torch.no_grad():
        for fpath in frame_files:
            try:
                image = Image.open(fpath).convert("RGB")
            except Exception:
                continue  # skip unreadable frame

            # prepare inputs
            inputs = processor(image, return_tensors="pt").to(device)

            # 2a) vision feature extraction
            vis_out = model.vision_model(pixel_values=inputs["pixel_values"])
            feat = vis_out.last_hidden_state.mean(dim=1)   # [1, hidden_dim]
            frame_feats.append(feat)

            # 2b) generate caption for this frame
            out_ids = model.generate(**inputs, max_length=max_length)
            caption = processor.decode(out_ids[0], skip_special_tokens=True)
            captions.append(caption)

    # If no audio given -> fallback to majority vote
    if not audio_path or not os.path.exists(audio_path):
        if len(captions) == 0:
            return ""
        return Counter(captions).most_common(1)[0][0]

    # 3) Extract simple audio vector (MFCC mean)
    try:
        mf = extract_audio_mfcc(audio_path, n_mfcc=40)
    except Exception:
        return Counter(captions).most_common(1)[0][0]
    audio_vec = torch.tensor(mf, dtype=torch.float32).unsqueeze(0).to(device)  # [1, n_mfcc]

    # 4) Project audio to vision dim and compute similarities
    hidden_dim = frame_feats[0].shape[1]
    fusion_proj = torch.nn.Linear(audio_vec.shape[1], hidden_dim).to(device)
    torch.nn.init.xavier_uniform_(fusion_proj.weight)

    with torch.no_grad():
        audio_emb = fusion_proj(audio_vec)  # [1, hidden_dim]

        caption_scores = {}
        for feat, cap in zip(frame_feats, captions):
            sim = F.cosine_similarity(audio_emb, feat, dim=1)  # tensor([score])
            sim_val = float(sim.item())
            caption_scores[cap] = caption_scores.get(cap, 0.0) + sim_val

    if len(caption_scores) == 0:
        return Counter(captions).most_common(1)[0][0]
    best_caption = max(caption_scores.items(), key=lambda x: x[1])[0]
    return best_caption

# Load dataset
dataset = pd.read_csv("data.csv")

# Store results
results = []

# Process each video in the dataset
for index, row in dataset.iterrows():
    video_path = row["video_path"].replace("\\", "/")  # Replace backslashes with forward slashes
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video name without extension

    # Paths for frames and audio
    frames_folder = os.path.join("frames", video_name)
    audio_folder = os.path.join("audio", video_name)
    audio_path = os.path.join(audio_folder, "audio.wav")

    # Generate one caption
    generated_caption = generate_video_caption(frames_folder, audio_path)

    # Save results
    results.append({
        "video_path": video_path,
        "generated_caption": generated_caption
    })

# Save generated captions to a separate CSV file
generated_captions_df = pd.DataFrame(results)
generated_captions_df.to_csv("generated_captions.csv", index=False)
print("Generated captions saved to generated_captions.csv")