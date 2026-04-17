import os
import cv2
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm

def create_masks():
    fg_dir = Path("raw_data/foregrounds")
    mask_dir = Path("raw_data/masks")
    mask_dir.mkdir(exist_ok=True)
    
    print("Generating masks from foreground patches...")
    files = list(fg_dir.glob("*.png"))
    for f in tqdm(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Simple adaptive thresholding for text extraction
        # Since text can be lighter or darker than background, we try to detect edges
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Dilate edges to create a solid mask for the text region
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)
        
        # Alternatively use Otsu + some morphological operations.
        # This is a heuristic for real TextZoom crops which don't have perfect ground-truth masks.
        cv2.imwrite(str(mask_dir / f.name), mask)

def download_backgrounds(num=20):
    bg_dir = Path("raw_data/backgrounds")
    bg_dir.mkdir(exist_ok=True)
    
    print(f"Downloading {num} random backgrounds...")
    for i in tqdm(range(num)):
        url = "https://picsum.photos/800/800"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(bg_dir / f"picsum_{i:03d}.jpg", "wb") as f:
                    f.write(resp.content)
        except Exception as e:
            print(f"Failed to download image {i}: {e}")

if __name__ == "__main__":
    create_masks()
    download_backgrounds()
