import os, glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from collections import Counter
from transformers import BlipProcessor, BlipForConditionalGeneration

def list_frames(folder, exts=("jpg","jpeg","png")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, f"*.{e}"))
    return sorted(files, key=lambda p: p.lower())

def sample_even(files, max_frames=3):  # 👈 set to 3 frames
    if len(files) <= max_frames:
        return files
    idx = np.linspace(0, len(files)-1, num=max_frames, dtype=int)
    return [files[i] for i in idx]

def caption_frame(path, processor, model, device, max_length=30):
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=False
        )
    return processor.decode(out[0], skip_special_tokens=True)

def caption_video(video_path, processor, model, device, frames_root="frames", max_frames=3):  # 👈 set to 3
    name = os.path.splitext(os.path.basename(video_path))[0]
    frames_folder = os.path.join(frames_root, name)
    files = list_frames(frames_folder)
    if not files:
        return ""
    files = sample_even(files, max_frames=max_frames)
    caps = [caption_frame(f, processor, model, device) for f in files]
    return Counter(caps).most_common(1)[0][0]  # majority vote

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device).eval()

    df = pd.read_csv("data.csv")
    rows = []
    for _, r in df.iterrows():
        vp = r["video_path"].replace("\\", "/")
        cap = caption_video(vp, processor, model, device, max_frames=3)  # 👈 set to 3
        rows.append({"video_path": vp, "generated_caption": cap})
        print(f"[ok] {os.path.basename(vp)} -> {cap}")

    out_path = "outputs/blip_large/generated_captions.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
