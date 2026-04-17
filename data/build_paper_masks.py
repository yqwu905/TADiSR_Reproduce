import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    print("Error: transformers library not found. Please install via 'pip install transformers'")
    exit(1)

class OCRMaskVerifier:
    def __init__(self, model_version="microsoft/trocr-small-printed", device="cpu"):
        print(f"Loading OCR model '{model_version}' to {device}...")
        
        # Determine device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.processor = TrOCRProcessor.from_pretrained(model_version)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_version).to(self.device)
        self.model.eval()
        
    def predict_text(self, image: Image.Image):
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
        
def generate_candidate_masks(img_gray):
    """
    Generates candidate segmentation masks relying on K-Means clustering
    extended with explicit border-polarity checks to eliminate inverted variants.
    """
    candidates = []
    
    # 1. K-Means pipeline
    pixels = img_gray.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    mask_kmeans = (labels.reshape(img_gray.shape) * 255).astype(np.uint8)
    
    # Border Polarity Guard (ensure mask text is white, bg is black)
    h, w = mask_kmeans.shape
    border_pixels = np.concatenate([mask_kmeans[0, :], mask_kmeans[-1, :], mask_kmeans[:, 0], mask_kmeans[:, -1]])
    if np.mean(border_pixels) > 127:
        mask_kmeans = 255 - mask_kmeans
    candidates.append(mask_kmeans)
    
    # 2. Morphological augmentation of kmeans mask to fix broken strokes
    kernel = np.ones((3, 3), np.uint8)
    mask_kmeans_closed = cv2.morphologyEx(mask_kmeans, cv2.MORPH_CLOSE, kernel)
    candidates.append(mask_kmeans_closed)
    
    # 3. Keep an adaptive thresholding method as a fallback
    mask_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    border_pixels_adapt = np.concatenate([mask_adapt[0, :], mask_adapt[-1, :], mask_adapt[:, 0], mask_adapt[:, -1]])
    if np.mean(border_pixels_adapt) > 127:
        mask_adapt = 255 - mask_adapt
    candidates.append(mask_adapt)
    
    return candidates

def clean_label(text):
    return text.upper().replace(" ", "").replace("-", "").replace("_", "")

def build_verified_masks(sample_limit=None):
    fg_dir = Path("raw_data/foregrounds")
    out_mask_dir = Path("raw_data/verified_masks")
    out_fg_dir = Path("raw_data/verified_foregrounds")
    
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_fg_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(fg_dir.glob("*.png"))
    if sample_limit:
        files = files[:sample_limit]
        
    print(f"Starting rigorous OCR-guided mask verification for {len(files)} foregrounds...")
    verifier = OCRMaskVerifier(device="mps") # Attempt Apple Silicon GPU naturally, fallbacks to CPU
    
    accepted = 0
    
    for f in tqdm(files):
        # Extract GT from filename (assumes format: label_index.png or label.png)
        stem = f.stem
        if "_" in stem:
            gt_text = stem.rsplit("_", 1)[0]
        else:
            gt_text = stem
            
        gt_clean = clean_label(gt_text)
        if len(gt_clean) < 2: 
            continue # skip extremely short non-texts or pure noise
            
        img_bgr = cv2.imread(str(f))
        if img_bgr is None:
            continue
            
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask_candidates = generate_candidate_masks(img_gray)
        
        best_mask = None
        
        # Test each mask via OCR
        for mask in mask_candidates:
            # Create a 3-channel version of mask to multiply
            mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Mask * Foreground: Keep only Text pixels, remainder is black
            masked_img = cv2.bitwise_and(img_bgr, mask_3c)
            
            # Convert to PIL for TrOCR
            pil_img = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
            
            # Predict
            pred_text = verifier.predict_text(pil_img)
            pred_clean = clean_label(pred_text)
            
            # Acceptance Protocol: Text-Aware Verifier matching GT
            if pred_clean == gt_clean:
                best_mask = mask
                break # Found a perfect semantic-preserving mask
                
        if best_mask is not None:
            # print(f"\n[ACCEPT] {f.name} | GT: {gt_clean} | Pred: {pred_clean}")
            # Save verified assets
            cv2.imwrite(str(out_mask_dir / f.name), best_mask)
            shutil.copy(str(f), str(out_fg_dir / f.name))
            accepted += 1
        else:
            # print(f"\n[REJECT] {f.name} | GT: {gt_clean} | Failed all candidates.")
            pass
            
    print(f"\nMask OCR filtering complete: {accepted}/{len(files)} masks accepted ({accepted/len(files)*100:.1f}%).")
    print(f"Verified assets stored in {out_mask_dir} & {out_fg_dir}")

def generate_sample_grid(num_samples=16):
    out_mask_dir = Path("raw_data/verified_masks")
    out_fg_dir = Path("raw_data/verified_foregrounds")
    
    files = list(out_mask_dir.glob("*.png"))[:num_samples]
    if not files:
        print("No verified masks to generate a grid for.")
        return
        
    grid_cols = 4
    grid_rows = (min(len(files), num_samples) + grid_cols - 1) // grid_cols
    
    cell_h, cell_w = 64, 128
    
    grid_img = np.zeros((grid_rows * (cell_h * 2 + 10), grid_cols * cell_w, 3), dtype=np.uint8)
    
    for idx, f in enumerate(files):
        r = idx // grid_cols
        c = idx % grid_cols
        
        y = r * (cell_h * 2 + 10)
        x = c * cell_w
        
        fg = cv2.imread(str(out_fg_dir / f.name))
        mask = cv2.imread(str(out_mask_dir / f.name), cv2.IMREAD_GRAYSCALE)
        
        if fg is None or mask is None:
            continue
            
        fg_resized = cv2.resize(fg, (cell_w, cell_h))
        mask_resized = cv2.resize(mask, (cell_w, cell_h))
        mask_3c = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        
        grid_img[y:y+cell_h, x:x+cell_w] = fg_resized
        grid_img[y+cell_h:y+cell_h*2, x:x+cell_w] = mask_3c
        
    cv2.imwrite("sample_masks.jpg", grid_img)
    print("Verification grid saved to sample_masks.jpg")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_evals", type=int, default=100, help="Number of files to evaluate initially")
    args = parser.parse_args()
    build_verified_masks(sample_limit=args.max_evals)
    generate_sample_grid()
