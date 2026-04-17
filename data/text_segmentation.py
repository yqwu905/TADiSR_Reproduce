#!/usr/bin/env python3
"""
Hi-SAM (SAM-TS) text segmentation wrapper.
Provides pixel-level text stroke segmentation for the FTSR pipeline.

Usage:
    from data.text_segmentation import TextSegmentor
    segmentor = TextSegmentor(device="cpu")
    mask = segmentor.segment(image_bgr)  # returns binary mask (0/255)
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISAM_DIR = ROOT / "third_party" / "Hi-SAM"


class TextSegmentor:
    """
    Text segmentation using Hi-SAM (SAM-TS) model.
    Falls back to a robust classical method if Hi-SAM is not available.
    """

    def __init__(self, device="cpu", model_type="vit_b"):
        self.device = device
        self.model = None
        self.use_hisam = False

        # Try loading Hi-SAM
        if HISAM_DIR.exists() and (HISAM_DIR / "hi_sam").exists():
            try:
                self._load_hisam(model_type)
                self.use_hisam = True
                print("[TextSegmentor] Hi-SAM loaded successfully.")
            except Exception as e:
                print(f"[TextSegmentor] Hi-SAM load failed: {e}")
                print("[TextSegmentor] Falling back to classical segmentation.")
        else:
            print(f"[TextSegmentor] Hi-SAM not found at {HISAM_DIR}")
            print("[TextSegmentor] Using classical segmentation (for local debugging).")
            print("[TextSegmentor] For production quality, set up Hi-SAM first.")

    def _load_hisam(self, model_type):
        """
        Load Hi-SAM model for text stroke segmentation.

        Fix D1: Hi-SAM has its own `SamTextPredictor` and hierarchical segmentation
        API, distinct from the generic SAM `SamPredictor`. We should NOT use
        grid-point prompts with generic SAM prediction — that produces random masks,
        not text segmentation masks.

        Hi-SAM models are fine-tuned on TextSeg and BTS datasets for text
        segmentation (paper Section 3.5: "We train SAM-TS using TextSeg and BTS").
        """
        import torch

        # Add Hi-SAM to path
        sys.path.insert(0, str(HISAM_DIR))

        # Hi-SAM provides its own model registry with text-aware heads
        from hi_sam.build_sam import sam_model_registry

        ckpt_map = {
            "vit_b": "sam_tss_b_hiertext.pth",
            "vit_l": "sam_tss_l_hiertext.pth",
            "vit_h": "sam_tss_h_hiertext.pth",
        }
        ckpt_name = ckpt_map.get(model_type, ckpt_map["vit_b"])
        ckpt_path = HISAM_DIR / "pretrained_checkpoint" / ckpt_name

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Determine device
        if self.device == "mps":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif self.device == "cuda":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Hi-SAM's sam_model_registry builds SAM with text-specific decoder heads
        sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
        sam.to(device)
        sam.eval()

        # Try to use Hi-SAM's text-specific predictor
        try:
            from hi_sam.predictor import SamHiTextPredictor
            self.predictor = SamHiTextPredictor(sam)
            self._use_text_predictor = True
            print("[TextSegmentor] Using SamHiTextPredictor (text-specific)")
        except ImportError:
            try:
                from hi_sam.predictor import SamPredictor
                self.predictor = SamPredictor(sam)
                self._use_text_predictor = False
                print("[TextSegmentor] Using SamPredictor (generic predictor)")
            except ImportError:
                raise ImportError("Cannot import predictor from Hi-SAM")

        self.model = sam
        self.torch_device = device

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Segment text regions in the image.

        Args:
            image_bgr: BGR image (OpenCV format)

        Returns:
            Binary mask (0=background, 255=text) same size as input
        """
        if self.use_hisam:
            return self._segment_hisam(image_bgr)
        else:
            return self._segment_classical(image_bgr)

    def _segment_hisam(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Segment using Hi-SAM model with text-specific prediction.

        Fix D1: Use Hi-SAM's text-aware API instead of generic point prompts.
        Hi-SAM's SAM-TS model has a dedicated text segmentation head that
        produces stroke-level text masks without requiring manual point prompts.
        """
        import torch

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]

        self.predictor.set_image(image_rgb)

        if self._use_text_predictor:
            # Hi-SAM text predictor: uses automatic text detection + seg
            try:
                # SamHiTextPredictor.predict() returns hierarchical text masks
                # at stroke/word/line levels
                masks, scores, logits = self.predictor.predict(
                    multimask_output=True,
                    target_level='stroke',  # stroke-level for per-character masks
                )

                # Combine high-confidence stroke masks
                combined = np.zeros((h, w), dtype=np.uint8)
                if isinstance(masks, np.ndarray) and masks.ndim == 3:
                    for i in range(masks.shape[0]):
                        if scores[i] > 0.5:
                            combined = np.maximum(
                                combined, (masks[i] * 255).astype(np.uint8)
                            )
                elif isinstance(masks, np.ndarray) and masks.ndim == 2:
                    combined = (masks * 255).astype(np.uint8)
                return combined

            except Exception as e:
                print(f"[TextSegmentor] SamHiTextPredictor.predict failed: {e}")
                print("[TextSegmentor] Falling back to automatic mask generation")

        # Fallback: use Hi-SAM's automatic mask generator if text predictor fails
        try:
            sys.path.insert(0, str(HISAM_DIR))
            from hi_sam.automatic_mask_generator import SamAutomaticMaskGenerator

            mask_gen = SamAutomaticMaskGenerator(
                model=self.model,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                min_mask_region_area=50,
            )
            masks_data = mask_gen.generate(image_rgb)

            combined = np.zeros((h, w), dtype=np.uint8)
            for md in masks_data:
                seg = md['segmentation']
                combined = np.maximum(combined, (seg * 255).astype(np.uint8))
            return combined

        except Exception as e:
            print(f"[TextSegmentor] Automatic mask generation failed: {e}")
            print("[TextSegmentor] Falling back to classical method")
            return self._segment_classical(image_bgr)

    def _segment_classical(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Robust classical text segmentation fallback.
        Uses Sauvola-style local adaptive thresholding which is specifically
        designed for document/text binarization.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Step 1: Sauvola-inspired local adaptive thresholding
        # Better than simple adaptive threshold for text
        k = 0.2  # Sauvola parameter
        window = max(15, min(h, w) // 4 | 1)  # ensure odd

        # Compute local mean and std using integral image
        mean = cv2.blur(gray.astype(np.float64), (window, window))
        sq_mean = cv2.blur((gray.astype(np.float64) ** 2), (window, window))
        std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

        threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
        binary = ((gray < threshold) * 255).astype(np.uint8)

        # Step 2: Border polarity check
        border = np.concatenate([
            binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
        ])
        if np.mean(border) > 127:
            binary = 255 - binary

        # Step 3: Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # Connect broken strokes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Step 4: Connected component filtering
        # Remove very small components (noise) and very large (background blobs)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        total_area = h * w
        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Keep components that are reasonable text size
            if 10 < area < total_area * 0.8:
                cleaned[labels == i] = 255

        return cleaned
