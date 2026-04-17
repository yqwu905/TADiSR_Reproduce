#!/usr/bin/env python3
"""
FTSR Dataset Construction - Main Pipeline Script

Reproduces the exact data synthesis pipeline from:
  "Text-Aware Real-World Image Super-Resolution via Diffusion Model
   with Joint Segmentation Decoders" (TADiSR, arXiv:2506.04641)

Pipeline steps (from Section 3.5 of the paper):
  1. Segment text in CTR images using SAM-TS (Hi-SAM)
  2. Filter using OCR consistency (original vs masked)
  3. Filter by long-edge/character-count ratio
  4. Super-resolve small/degraded patches with HAT (Issue 8 fix)
  5. Filter LSDIR backgrounds to exclude images containing text (Issue 9 fix)
  6. Paste filtered text patches onto LSDIR backgrounds
  7. Merge corresponding segmentation masks
  8. Apply Real-ESRGAN degradation (two-stage) to generate LR images
  9. Output (x_L, x_H, s) triplets

Usage:
    python data/build_ftsr.py --num_samples 100
"""
import os
import sys
import cv2
import random
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.text_segmentation import TextSegmentor
from data.ocr_filter import OCRFilter
from data.degradation import RealESRGANDegradation


def load_foreground_images(fg_dir: Path, max_count: int = None):
    """Load text foreground images and their GT labels from filename."""
    images = []
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted([f for f in fg_dir.iterdir() if f.suffix.lower() in exts])
    if max_count:
        files = files[:max_count]

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        # Extract GT label from filename (format: label_index.png)
        stem = f.stem
        parts = stem.rsplit("_", 1)
        label = parts[0] if len(parts) > 1 else stem
        images.append({"path": f, "image": img, "label": label})

    return images


def load_backgrounds(bg_dir: Path, max_count: int = None):
    """Load background images from LSDIR or similar."""
    bgs = []
    exts = {'.png', '.jpg', '.jpeg'}
    files = sorted([f for f in bg_dir.iterdir() if f.suffix.lower() in exts])
    if max_count:
        files = files[:max_count]

    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            bgs.append(img)

    return bgs


def filter_by_ratio(items, min_ratio=3.0):
    """
    Paper: "filter out CTR samples with a small long-edge/character count ratio"
    This removes patches where text is too densely packed relative to image size.
    """
    filtered = []
    for item in items:
        h, w = item["image"].shape[:2]
        long_edge = max(h, w)
        char_count = max(len(item["label"]), 1)
        ratio = long_edge / char_count
        if ratio >= min_ratio:
            filtered.append(item)
    return filtered


# ---------------------------------------------------------------------------
# Issue 8: HAT super-resolution for text patches
# ---------------------------------------------------------------------------
def super_resolve_patches(items, device="cpu"):
    """
    Paper (Section 3.5): "super-resolve them with SOTA Real-SR methods [HAT]"

    Apply HAT (or a configured SR model) to enhance small/degraded text patches
    before compositing onto backgrounds.

    Falls back to bicubic upscaling if HAT is not available.
    """
    hat_available = False

    # Try loading HAT
    try:
        # HAT from: https://github.com/XPixelGroup/HAT
        sys.path.insert(0, str(ROOT / "third_party" / "HAT"))
        from hat.archs.hat_arch import HAT
        import torch

        hat_model_path = ROOT / "third_party" / "HAT" / "pretrained" / "HAT_SRx4.pth"
        if hat_model_path.exists():
            print(f"  [HAT] Loading model from {hat_model_path}")
            hat_model = HAT(
                upscale=4, in_chans=3, img_size=64,
                window_size=16, compress_ratio=3,
                squeeze_factor=30, conv_scale=0.01,
                overlap_ratio=0.5, img_range=1.,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle',
                resi_connection='1conv',
            )
            state_dict = torch.load(str(hat_model_path), map_location="cpu")
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]
            hat_model.load_state_dict(state_dict, strict=False)

            if device == "cuda" and torch.cuda.is_available():
                hat_model = hat_model.cuda()
            hat_model.eval()
            hat_available = True
            print(f"  [HAT] Model loaded successfully")
        else:
            print(f"  [HAT] Model weights not found at {hat_model_path}")
    except Exception as e:
        print(f"  [HAT] Could not load HAT: {e}")

    if not hat_available:
        print(f"  [HAT] Falling back to bicubic 2× upscaling")
        print(f"  [HAT] For paper-faithful results, set up HAT in third_party/HAT/")

    enhanced = []
    for item in tqdm(items, desc="Super-resolving"):
        img = item["image"]
        mask = item["mask"]
        h, w = img.shape[:2]

        # Only super-resolve small patches (long edge < 128px)
        if max(h, w) < 128:
            if hat_available:
                import torch
                # Prepare for HAT: BGR→RGB, HWC→CHW, normalize
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
                if device == "cuda":
                    tensor = tensor.cuda()
                with torch.no_grad():
                    sr = hat_model(tensor)
                sr = sr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                sr = np.clip(sr * 255, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

                # Also upscale mask to match
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            else:
                # Bicubic 4× upscale fallback (matching HAT's 4× SR scale)
                scale = 4
                img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(mask, (w * scale, h * scale),
                                   interpolation=cv2.INTER_NEAREST)

        item["image"] = img
        item["mask"] = mask
        enhanced.append(item)

    return enhanced


# ---------------------------------------------------------------------------
# Issue 9: LSDIR background OCR filtering
# ---------------------------------------------------------------------------
def filter_backgrounds_ocr(backgrounds, ocr_filter, max_check=None):
    """
    Paper (Section 3.5): "to avoid label ambiguity, we exclude LSDIR samples
    that contain text by using an OCR-based filtering step"

    Remove background images that contain any detectable text.
    """
    filtered = []
    total = len(backgrounds)
    check_count = min(total, max_check) if max_check else total

    for i, bg in enumerate(tqdm(backgrounds[:check_count], desc="Filtering BG text")):
        if not ocr_filter.detect_has_text(bg):
            filtered.append(bg)

    # Append unchecked backgrounds if max_check was set
    if max_check and max_check < total:
        filtered.extend(backgrounds[max_check:])

    removed = total - len(filtered)
    print(f"  Background OCR filter: removed {removed}/{check_count} images with text")
    return filtered


def compose_image(bg_img: np.ndarray, fg_items: list, canvas_size=512):
    """
    Compose: paste text patches onto background, merge masks.

    Paper: "a random number of cropped images are pasted onto high-quality
    HR samples from the LSDIR dataset, with the corresponding text
    segmentation maps placed on a zero-initialized image of the same size"
    """
    # Resize background to canvas size
    bg = cv2.resize(bg_img, (canvas_size, canvas_size))
    hr_image = bg.copy()
    mask_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    for item in fg_items:
        fg = item["image"]
        fg_mask = item["mask"]
        h, w = fg.shape[:2]

        # Random scale (ensure it fits in canvas)
        max_dim = max(h, w)
        if max_dim > canvas_size * 0.8:
            scale = (canvas_size * 0.4) / max_dim
        else:
            scale = random.uniform(0.5, 1.5)
            # Clamp to fit canvas
            new_h = int(h * scale)
            new_w = int(w * scale)
            if new_h >= canvas_size or new_w >= canvas_size:
                scale = min(canvas_size * 0.8 / h, canvas_size * 0.8 / w)

        new_h = max(int(h * scale), 1)
        new_w = max(int(w * scale), 1)

        fg_resized = cv2.resize(fg, (new_w, new_h))
        mask_resized = cv2.resize(fg_mask, (new_w, new_h),
                                  interpolation=cv2.INTER_NEAREST)

        # Random position
        max_y = max(canvas_size - new_h, 1)
        max_x = max(canvas_size - new_w, 1)
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)

        # Alpha blend: paste foreground where mask is active
        mask_3c = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        roi = hr_image[y:y+new_h, x:x+new_w].astype(np.float32)
        blended = roi * (1 - mask_3c) + fg_resized.astype(np.float32) * mask_3c
        hr_image[y:y+new_h, x:x+new_w] = blended.astype(np.uint8)

        # Merge mask
        mask_canvas[y:y+new_h, x:x+new_w] = np.maximum(
            mask_canvas[y:y+new_h, x:x+new_w], mask_resized
        )

    return hr_image, mask_canvas


def generate_verification_grid(hr_dir: Path, mask_dir: Path, lr_dir: Path,
                                output_path: Path, num_samples=12):
    """Generate a visual verification grid showing HR, Mask, LR triplets."""
    files = sorted(list(hr_dir.glob("*.png")))[:num_samples]
    if not files:
        print("[Verify] No files to visualize.")
        return

    cols = min(4, len(files))
    rows = (len(files) + cols - 1) // cols

    cell_w, cell_h = 192, 192
    gap = 4
    grid_w = cols * cell_w + (cols - 1) * gap
    grid_h = rows * (cell_h * 3 + gap * 2) + (rows - 1) * gap * 2

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark bg

    for idx, f in enumerate(files):
        r = idx // cols
        c = idx % cols
        x = c * (cell_w + gap)
        y = r * (cell_h * 3 + gap * 4)

        hr = cv2.imread(str(hr_dir / f.name))
        mask = cv2.imread(str(mask_dir / f.name), cv2.IMREAD_GRAYSCALE)
        lr = cv2.imread(str(lr_dir / f.name))

        if hr is None:
            continue

        hr_small = cv2.resize(hr, (cell_w, cell_h))
        grid[y:y+cell_h, x:x+cell_w] = hr_small

        if mask is not None:
            mask_3c = cv2.cvtColor(cv2.resize(mask, (cell_w, cell_h)),
                                   cv2.COLOR_GRAY2BGR)
            grid[y+cell_h+gap:y+cell_h*2+gap, x:x+cell_w] = mask_3c

        if lr is not None:
            lr_up = cv2.resize(lr, (cell_w, cell_h))
            grid[y+cell_h*2+gap*2:y+cell_h*3+gap*2, x:x+cell_w] = lr_up

    cv2.imwrite(str(output_path), grid)
    print(f"[Verify] Grid saved to {output_path}")


def build_ftsr(args):
    """Main FTSR construction pipeline."""
    fg_dir = Path(args.fg_dir)
    bg_dir = Path(args.bg_dir)
    output_dir = Path(args.output_dir)

    hr_dir = output_dir / "HR"
    lr_dir = output_dir / "LR"
    mask_dir = output_dir / "Mask"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FTSR Dataset Construction Pipeline")
    print("=" * 60)

    # ---- Step 1: Load foreground text patches ----
    print(f"\n--- Step 1: Loading foreground text images from {fg_dir} ---")
    fg_items = load_foreground_images(fg_dir, max_count=args.max_fg)
    print(f"  Loaded {len(fg_items)} text patches")

    if not fg_items:
        print("ERROR: No foreground images found. Run download_datasets.py first.")
        return

    # ---- Step 2: Text segmentation (SAM-TS / Hi-SAM) ----
    print(f"\n--- Step 2: Text segmentation (SAM-TS) ---")
    segmentor = TextSegmentor(device=args.device)

    segmented = []
    for item in tqdm(fg_items, desc="Segmenting"):
        mask = segmentor.segment(item["image"])
        # Check mask is non-trivial (has some text pixels)
        text_ratio = np.sum(mask > 127) / mask.size
        if 0.01 < text_ratio < 0.95:  # Neither empty nor all-white
            item["mask"] = mask
            segmented.append(item)

    print(f"  Segmented: {len(segmented)}/{len(fg_items)} valid masks")

    # ---- Step 3: OCR filtering ----
    print(f"\n--- Step 3: OCR consistency filtering ---")
    ocr = None
    if args.skip_ocr:
        print("  Skipping OCR filter (--skip_ocr)")
        verified = segmented
    else:
        try:
            ocr = OCRFilter(device=args.device)
            verified = []
            for item in tqdm(segmented, desc="OCR filtering"):
                accepted, ocr_orig, ocr_masked, score = ocr.verify_mask(
                    item["image"], item["mask"], gt_label=item["label"]
                )
                if accepted:
                    verified.append(item)
            print(f"  OCR verified: {len(verified)}/{len(segmented)} accepted")
        except Exception as e:
            print(f"  OCR filter error: {e}")
            print(f"  Continuing without OCR filtering.")
            verified = segmented

    if not verified:
        print("ERROR: No images passed filtering. Using all segmented images.")
        verified = segmented

    # ---- Step 4: Filter by long-edge/char ratio ----
    print(f"\n--- Step 4: Ratio filtering ---")
    verified = filter_by_ratio(verified, min_ratio=args.min_ratio)
    print(f"  After ratio filter: {len(verified)} patches")

    # ---- Step 4b: HAT super-resolution for small patches (Issue 8) ----
    if not args.skip_hat:
        print(f"\n--- Step 4b: Super-resolving small text patches (HAT) ---")
        verified = super_resolve_patches(verified, device=args.device)
        print(f"  After HAT: {len(verified)} patches enhanced")
    else:
        print(f"\n--- Step 4b: Skipping HAT super-resolution (--skip_hat) ---")

    # ---- Step 5: Load backgrounds ----
    print(f"\n--- Step 5: Loading backgrounds from {bg_dir} ---")
    backgrounds = load_backgrounds(bg_dir, max_count=args.max_bg)
    print(f"  Loaded {len(backgrounds)} backgrounds")

    if not backgrounds:
        print("ERROR: No background images found. Run download_datasets.py first.")
        return

    # ---- Step 5b: Filter backgrounds with OCR (Issue 9) ----
    if not args.skip_bg_ocr:
        print(f"\n--- Step 5b: Filtering backgrounds to exclude text (OCR) ---")
        if ocr is None:
            try:
                ocr = OCRFilter(device=args.device)
            except Exception as e:
                print(f"  Could not initialize OCR for background filtering: {e}")
                ocr = None
        if ocr is not None:
            backgrounds = filter_backgrounds_ocr(
                backgrounds, ocr, max_check=args.max_bg_ocr_check
            )
        else:
            print("  Skipping background OCR filter (OCR unavailable)")
    else:
        print(f"\n--- Step 5b: Skipping background OCR filter (--skip_bg_ocr) ---")

    print(f"  Using {len(backgrounds)} backgrounds for composition")

    # ---- Step 6: Compose HR images + masks ----
    print(f"\n--- Step 6: Composing {args.num_samples} HR images + masks ---")
    for i in tqdm(range(args.num_samples), desc="Composing"):
        bg = random.choice(backgrounds)
        # Random 1-5 text patches per image
        n_patches = random.randint(1, min(5, len(verified)))
        selected = random.sample(verified, n_patches)

        hr_img, mask_img = compose_image(bg, selected, canvas_size=args.canvas_size)

        fname = f"ftsr_{i:06d}.png"
        cv2.imwrite(str(hr_dir / fname), hr_img)
        cv2.imwrite(str(mask_dir / fname), mask_img)

    print(f"  Composed {args.num_samples} HR + Mask pairs")

    # ---- Step 7: Real-ESRGAN degradation (two-stage) ----
    print(f"\n--- Step 7: Applying Real-ESRGAN degradation (two-stage, SF={args.sf}) ---")
    degrader = RealESRGANDegradation(scale_factor=args.sf)

    hr_files = sorted(list(hr_dir.glob("*.png")))
    for f in tqdm(hr_files, desc="Degrading"):
        hr = cv2.imread(str(f))
        if hr is None:
            continue
        lr = degrader.degrade(hr)
        cv2.imwrite(str(lr_dir / f.name), lr)

    print(f"  Generated {len(hr_files)} LR images")

    # ---- Step 8: Verification ----
    print(f"\n--- Step 8: Verification ---")
    grid_path = ROOT / "ftsr_verification.jpg"
    generate_verification_grid(hr_dir, mask_dir, lr_dir, grid_path)

    # Count and report
    hr_count = len(list(hr_dir.glob("*.png")))
    lr_count = len(list(lr_dir.glob("*.png")))
    mask_count = len(list(mask_dir.glob("*.png")))
    print(f"\n{'=' * 60}")
    print(f"FTSR Dataset Complete!")
    print(f"  HR:   {hr_count} images in {hr_dir}")
    print(f"  LR:   {lr_count} images in {lr_dir}")
    print(f"  Mask: {mask_count} images in {mask_dir}")
    print(f"  Verification grid: {grid_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FTSR dataset (TADiSR paper pipeline)"
    )

    # Data paths
    parser.add_argument("--fg_dir", type=str,
                       default=str(ROOT / "raw_data" / "foregrounds"),
                       help="Directory with text foreground patches")
    parser.add_argument("--bg_dir", type=str,
                       default=str(ROOT / "raw_data" / "LSDIR"),
                       help="Directory with background images (LSDIR)")
    parser.add_argument("--output_dir", type=str,
                       default=str(ROOT / "dataset" / "FTSR"),
                       help="Output directory for FTSR dataset")

    # Pipeline parameters
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of FTSR samples to generate")
    parser.add_argument("--canvas_size", type=int, default=512,
                       help="HR image canvas size")
    parser.add_argument("--sf", type=int, default=4,
                       help="Super-resolution scale factor")
    parser.add_argument("--max_fg", type=int, default=None,
                       help="Max foreground images to load")
    parser.add_argument("--max_bg", type=int, default=None,
                       help="Max background images to load")
    parser.add_argument("--min_ratio", type=float, default=2.0,
                       help="Minimum long-edge/char-count ratio")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "mps", "cuda"],
                       help="Device for segmentation and OCR models")

    # Skip flags
    parser.add_argument("--skip_ocr", action="store_true",
                       help="Skip OCR filtering step (for faster debugging)")
    parser.add_argument("--skip_hat", action="store_true",
                       help="Skip HAT super-resolution step")
    parser.add_argument("--skip_bg_ocr", action="store_true",
                       help="Skip LSDIR background OCR text filtering")
    parser.add_argument("--max_bg_ocr_check", type=int, default=None,
                       help="Max backgrounds to OCR-check (for speed)")

    args = parser.parse_args()
    build_ftsr(args)
