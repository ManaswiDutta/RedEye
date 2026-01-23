import os
import json
import csv

ROOT = "data/raw"         # adjust if needed
OUT_CSV = "dataset_auto.csv"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "ups", "downs", "score", "title", "subreddit"])

    for root, dirs, files in os.walk(ROOT):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(root, file)
                json_path = image_path + ".json"
                
                if not os.path.exists(json_path):
                    # Check if maybe it's just filename.json (older format?)
                    # But based on the file listing, it's image.ext.json
                    continue
                
                try:
                    with open(json_path, "r", encoding="utf-8") as meta:
                        data = json.load(meta)
                    
                    ups = data.get("ups", 0)
                    downs = data.get("downs", 0)
                    score = data.get("score", 0)
                    title = data.get("title", "")
                    subreddit = data.get("subreddit", "")
                    
                    writer.writerow([image_path, ups, downs, score, title, subreddit])
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {json_path}: {e}")

print("CSV generated:", OUT_CSV)
