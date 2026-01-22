import joblib
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import sys

MODEL_PATH = "models/regressor.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, clip, processor

def predict(image_path):
    model, clip, processor = load_model()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        emb = clip.get_image_features(**inputs)[0].cpu().numpy()

    emb = emb.reshape(1, -1)  # make batch shape
    preds = model.predict(emb)[0] # Returns [ups, downs, score]
    return preds

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    preds = predict(img_path)
    
    print(f"\nPredictions for '{img_path}':")
    print(f"  Ups:   {preds[0]:.1f}")
    print(f"  Downs: {preds[1]:.1f}")
    print(f"  Score: {preds[2]:.1f}")
