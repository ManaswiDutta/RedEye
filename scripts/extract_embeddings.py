import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

LABELS_CSV = "data/labels.csv"
IMAGES_DIR = "data/images/"
OUT_EMB = "data/embeddings/embeddings.npy"
OUT_LABELS = "data/embeddings/labels.npy"

os.makedirs("data/embeddings/", exist_ok=True)

def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def main():
    df = pd.read_csv(LABELS_CSV)
    model, processor = load_clip()
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
            img_path = os.path.join(IMAGES_DIR, row["filename"])
            score = float(row["score"])

            try:
                image = Image.open(img_path).convert("RGB")
            except:
                print(f"[WARN] failed to open {img_path}")
                continue

            inputs = processor(images=image, return_tensors="pt")
            outputs = model.get_image_features(**inputs)

            vec = outputs[0].cpu().numpy()
            embeddings.append(vec)
            labels.append(score)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    np.save(OUT_EMB, embeddings)
    np.save(OUT_LABELS, labels)

    print(f"Saved embeddings to {OUT_EMB}")
    print(f"Saved labels to {OUT_LABELS}")
    print("Done!")

if __name__ == "__main__":
    main()
