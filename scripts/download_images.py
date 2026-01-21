import pandas as pd
import requests
import os
import time
from urllib.parse import urlparse
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RAW_CSV = os.environ.get("RAW_CSV", "data/raw/reddata.csv")
OUT_DIR = "data/images/"
LABELS_OUT = "data/labels.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# Central session with retries and realistic headers
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*"
})

def extract_image_url(post_url):
    """
    Extracts the direct image URL from a Reddit post using its .json endpoint.
    """
    try:
        # Clean the URL and append .json
        clean_url = post_url.split('#')[0].rstrip('/')
        json_url = f"{clean_url}.json"
        
        r = session.get(json_url, timeout=15)
        if r.status_code != 200:
            return None
            
        data = r.json()
        
        # Reddit JSON for a single post is usually a list of two objects (post, comments)
        if isinstance(data, list) and len(data) > 0:
            post_data = data[0]['data']['children'][0]['data']
            
            # 1. Direct image link (i.redd.it, imgur, etc.)
            url = post_data.get('url', '')
            if url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                return url
                
            # 2. Check for gallery
            if 'is_gallery' in post_data and post_data['is_gallery']:
                # Get the first image in the gallery
                items = post_data.get('gallery_data', {}).get('items', [])
                if items:
                    media_id = items[0]['media_id']
                    # Construct URL based on media ID
                    return f"https://i.redd.it/{media_id}.jpg"

            # 3. Check previews for the primary source
            previews = post_data.get('preview', {}).get('images', [])
            if previews:
                return previews[0]['source']['url'].replace('&amp;', '&')

        return None
    except Exception as e:
        print(f"[ERROR] JSON extraction failed for {post_url}: {e}")
        return None

def download_file(url, dest_path):
    """Downloads a file to the destination path."""
    try:
        r = session.get(url, stream=True, timeout=20)
        if r.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Download failed for {url}: {e}")
        return False

def main():
    if not os.path.exists(RAW_CSV):
        print(f"Error: {RAW_CSV} not found!")
        return

    df = pd.read_csv(RAW_CSV)
    labels = []

    print(f"Starting download for {len(df)} posts...")

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        post_url = row["url"]
        score = row["score"]

        img_url = extract_image_url(post_url)
        if not img_url:
            # Fallback regex if JSON failed but we have a direct link in the URL column possibly
            if str(post_url).lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                img_url = post_url
            else:
                continue

        # create safe filename
        parsed = urlparse(img_url)
        basename = os.path.basename(parsed.path)
        if not basename or '.' not in basename:
            ext = '.jpg' # default
            if 'png' in img_url.lower(): ext = '.png'
            basename = f"img_{i}{ext}"

        save_path = os.path.join(OUT_DIR, basename)

        if download_file(img_url, save_path):
            labels.append((basename, score))
        
        # Polite delay to avoid rate limiting
        time.sleep(0.5)

    # save labels csv
    if labels:
        out_df = pd.DataFrame(labels, columns=["filename", "score"])
        out_df.to_csv(LABELS_OUT, index=False)
        print(f"\nSuccess! Saved {len(labels)} labels to {LABELS_OUT}")
    else:
        print("\nNo images were downloaded. Please check the logs.")

if __name__ == "__main__":
    main()
