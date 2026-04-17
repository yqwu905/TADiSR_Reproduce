import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def get_robust_mask(img_str_path):
    img = cv2.imread(img_str_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 1. Morphological Gradient to highlight text strokes accurately
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    
    # 2. Binarize the gradient using Otsu
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Connect components
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    
    # 4. Alternatively, use k-means on the original image for polarity-aware segmentation
    pixels = img.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    mask = (labels.reshape(img.shape) * 255).astype(np.uint8)
    
    # Check polarity: assuming text does not touch the majority of the border
    h, w = mask.shape
    border_pixels = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if np.mean(border_pixels) > 127:
        mask = 255 - mask # Invert so background is black
        
    return mask, connected

files = list(Path("raw_data/foregrounds").glob("*.png"))[:10]
for f in files:
    mask_kmeans, mask_grad = get_robust_mask(str(f))
    
    # Print small ascii representation of kmeans mask
    small = cv2.resize(mask_kmeans, (40, 10))
    print(f"\n--- {f.name} (KMEANS) ---")
    for row in small:
        print("".join(["#" if p > 127 else "." for p in row]))
        
    small2 = cv2.resize(mask_grad, (40, 10))
    print(f"--- {f.name} (GRADIENT) ---")
    for row in small2:
        print("".join(["#" if p > 127 else "." for p in row]))
