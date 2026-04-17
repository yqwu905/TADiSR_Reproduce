import os
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class TADiSRDataSynthesizer:
    def __init__(self, fg_dir, mask_dir, bg_dir, output_dir, output_size=(512, 512)):
        """
        fg_dir: Directory containing foreground text patches
        mask_dir: Directory containing foreground text segmentation masks (1-channel or 3-channel binary)
        bg_dir: Directory containing high-quality background images (text-free)
        output_dir: Where to save the synthesized HR images and merged masks
        """
        self.fg_dir = Path(fg_dir)
        self.mask_dir = Path(mask_dir)
        self.bg_dir = Path(bg_dir)
        self.output_dir = Path(output_dir)
        
        self.hr_out = self.output_dir / "HR"
        self.mask_out = self.output_dir / "Mask"
        self.hr_out.mkdir(parents=True, exist_ok=True)
        self.mask_out.mkdir(parents=True, exist_ok=True)
        
        self.output_size = output_size
        
        self.bgs = list(self.bg_dir.glob("*.png")) + list(self.bg_dir.glob("*.jpg"))
        self.fgs = list(self.fg_dir.glob("*.png")) + list(self.fg_dir.glob("*.jpg"))
        
        if len(self.bgs) == 0:
            print("Warning: No background images found.")
        if len(self.fgs) == 0:
            print("Warning: No foreground images found.")

    def random_transform(self, img, mask):
        """
        Apply random scaling and small rotation to the foreground and mask.
        """
        h, w = img.shape[:2]
        
        # Scale: 0.5 to 1.5
        scale = random.uniform(0.5, 1.5)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0:
            return img, mask
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Roation: -15 to 15 degrees
        angle = random.uniform(-15.0, 15.0)
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1)
        
        img = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        mask = cv2.warpAffine(mask, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return img, mask

    def composite(self, bg_img, fg_patches):
        """
        Paste fg_patches onto bg_img. 
        Returns composite_img, combined_mask.
        """
        bg_h, bg_w = bg_img.shape[:2]
        comp_img = bg_img.copy()
        comp_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        
        for (fg_img, fg_mask) in fg_patches:
            fg_img, fg_mask = self.random_transform(fg_img, fg_mask)
            fh, fw = fg_img.shape[:2]
            
            if fh >= bg_h or fw >= bg_w:
                continue
                
            x_min = random.randint(0, bg_w - fw)
            y_min = random.randint(0, bg_h - fh)
            
            alpha = (fg_mask > 0).astype(np.float32)
            if len(alpha.shape) == 2:
                alpha = np.expand_dims(alpha, axis=-1)
                
            roi_img = comp_img[y_min:y_min+fh, x_min:x_min+fw]
            
            # Blend
            blended = roi_img * (1 - alpha) + fg_img * alpha
            comp_img[y_min:y_min+fh, x_min:x_min+fw] = blended
            
            # Mask accumulation
            roi_mask = comp_mask[y_min:y_min+fh, x_min:x_min+fw]
            roi_mask = np.maximum(roi_mask, fg_mask)
            comp_mask[y_min:y_min+fh, x_min:x_min+fw] = roi_mask
            
        return comp_img, comp_mask
        
    def generate_samples(self, n_samples):
        print(f"Generating {n_samples} composite samples...")
        for i in tqdm(range(n_samples)):
            if len(self.bgs) == 0 or len(self.fgs) == 0:
                print("Missing data to composite.")
                break
                
            bg_path = random.choice(self.bgs)
            bg = cv2.imread(str(bg_path))
            if bg is None: continue
            
            bg = cv2.resize(bg, self.output_size)
            
            # Select 1 ~ 5 foreground patches
            num_fgs = random.randint(1, 5)
            patches = []
            for _ in range(num_fgs):
                fg_path = random.choice(self.fgs)
                fg = cv2.imread(str(fg_path))
                
                # Fetch corresponding mask
                mask_path = self.mask_dir / fg_path.name
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if fg is not None and mask is not None:
                    # Convert mask to 0/255 if not already
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    patches.append((fg, mask))
                    
            hr_img, hr_mask = self.composite(bg, patches)
            
            filename = f"synth_{i:06d}.png"
            cv2.imwrite(str(self.hr_out / filename), hr_img)
            cv2.imwrite(str(self.mask_out / filename), hr_mask)

if __name__ == "__main__":
    # Example Offline Generation Pipeline Runner
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_dir", type=str, default="raw_data/verified_foregrounds", help="Path to text foregrounds")
    parser.add_argument("--mask_dir", type=str, default="raw_data/verified_masks", help="Path to text masks")
    parser.add_argument("--bg_dir", type=str, default="raw_data/backgrounds", help="Path to text-free backgrounds")
    parser.add_argument("--output_dir", type=str, default="dataset/FTSR", help="Output dir for FTSR dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of HR samples to compose")
    args = parser.parse_args()
    
    synth = TADiSRDataSynthesizer(
        fg_dir=args.fg_dir,
        mask_dir=args.mask_dir,
        bg_dir=args.bg_dir,
        output_dir=args.output_dir
    )
    synth.generate_samples(args.num_samples)
