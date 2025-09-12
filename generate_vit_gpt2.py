import os, glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from collections import Counter
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def list_frames(folder, exts=("jpg","jpeg","png")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, f"*.{e}"))
    return sorted(files, key=lambda p: p.lower())

def sample_even(files, max_frames=3):  # ✅ 3 frames only
    if len(files) <= max_frames:
        return files
    idx = np.linspace(0, len(files)-1, num=max_frames, dtype=int)
    return [files[i] for i in idx]

def caption_frame(path, feature_extractor, tokenizer, model, device, max_length=30):
    img = Image.open(path).convert("RGB")
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        out = model.generate(pixel_values, max_length=max_length, num_beams=5)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def caption_video(video_path, feature_extractor, tokenizer, model, device, frames_root="frames", max_frames=3):
    name = os.path.splitext(os.path.basename(video_path))[0]
    frames_folder = os.path.join(frames_root, name)
    files = list_frames(frames_folder)
    if not files:
        return ""
    files = sample_even(files, max_frames=max_frames)
    caps = [caption_frame(f, feature_extractor, tokenizer, model, device) for f in files]
    return Counter(caps).most_common(1)[0][0]  # majority vote

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device).eval()
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    df = pd.read_csv("data.csv")
    rows = []
    for _, r in df.iterrows():
        vp = r["video_path"].replace("\\", "/")
        cap = caption_video(vp, feature_extractor, tokenizer, model, device, max_frames=3)
        rows.append({"video_path": vp, "generated_caption": cap})
        print(f"[ok] {os.path.basename(vp)} -> {cap}")

    out_path = "outputs/vit_gpt2/generated_captions.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

