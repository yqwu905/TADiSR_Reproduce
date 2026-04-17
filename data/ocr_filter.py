#!/usr/bin/env python3
"""
OCR-based mask quality filter for the FTSR pipeline.

Per the TADiSR paper (Section 3.5):
  "filter reliable pseudo-ground-truth samples by comparing OCR-based
   text recognition results between the original images and their
   segmentation maps"

Fix applied:
  - Issue 10: Added PP-OCR (PaddleOCR) as primary OCR engine for Chinese+English
              support, with TrOCR as fallback for English-only environments.

The paper uses PP-OCR [du2020pp] which supports Chinese text — critical
because CTR is a Chinese Text Recognition dataset.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# --- Priority 1: PP-OCR (paper's choice, supports Chinese) ---
try:
    from paddleocr import PaddleOCR
    HAS_PPOCR = True
except ImportError:
    HAS_PPOCR = False

# --- Priority 2: TrOCR (English-only fallback) ---
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    HAS_TROCR = True
except ImportError:
    HAS_TROCR = False


class OCRFilter:
    """
    Filters text patches by comparing OCR results on original vs masked images.
    Only patches where the mask preserves OCR readability are accepted.

    Priority:
      1. PaddleOCR (PP-OCR) — paper's choice, supports Chinese+English
      2. TrOCR — English-only fallback
    """

    def __init__(self, device="cpu", model_name="microsoft/trocr-small-printed",
                 lang="ch", use_ppocr=None):
        """
        Args:
            device: "cpu", "cuda", or "mps"
            model_name: TrOCR model name (used only as fallback)
            lang: PaddleOCR language ("ch" for Chinese, "en" for English)
            use_ppocr: Force PP-OCR (True), force TrOCR (False), or auto-detect (None)
        """
        self.ocr_engine = None
        self._engine_type = None

        # Auto-detect available engine
        if use_ppocr is None:
            use_ppocr = HAS_PPOCR

        if use_ppocr and HAS_PPOCR:
            self._init_ppocr(lang)
        elif HAS_TROCR:
            self._init_trocr(device, model_name)
        else:
            raise ImportError(
                "No OCR engine available. Install one of:\n"
                "  pip install paddlepaddle paddleocr   (recommended, Chinese+English)\n"
                "  pip install transformers              (English-only fallback)"
            )

    def _init_ppocr(self, lang):
        """Initialize PP-OCR engine (paper's choice)."""
        print(f"[OCRFilter] Using PP-OCR (lang={lang}) — matches paper's OCR engine")
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            use_gpu=True,
        )
        self._engine_type = "ppocr"

    def _init_trocr(self, device, model_name):
        """Initialize TrOCR as fallback."""
        print(f"[OCRFilter] PP-OCR not available. Using TrOCR fallback (English only).")
        print(f"[OCRFilter] WARNING: TrOCR cannot recognize Chinese text (CTR dataset)!")
        print(f"[OCRFilter] For paper-faithful results, install paddleocr.")

        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._engine_type = "trocr"

    def recognize(self, image_bgr: np.ndarray) -> str:
        """Run OCR on a BGR image and return predicted text."""
        if self._engine_type == "ppocr":
            return self._recognize_ppocr(image_bgr)
        else:
            return self._recognize_trocr(image_bgr)

    def _recognize_ppocr(self, image_bgr: np.ndarray) -> str:
        """PP-OCR recognition."""
        result = self.ocr_engine.ocr(image_bgr, cls=True)
        if result is None or len(result) == 0:
            return ""
        # Concatenate all detected text lines
        texts = []
        for line in result:
            if line is None:
                continue
            for item in line:
                if item and len(item) >= 2:
                    texts.append(item[1][0])  # (text, confidence)
        return " ".join(texts).strip()

    def _recognize_trocr(self, image_bgr: np.ndarray) -> str:
        """TrOCR recognition."""
        pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        pixel_values = self.processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            ids = self.model.generate(pixel_values, max_new_tokens=30)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip()

    def detect_has_text(self, image_bgr: np.ndarray) -> bool:
        """
        Check if an image contains any text (used for LSDIR background filtering).
        Returns True if text is detected.
        """
        if self._engine_type == "ppocr":
            result = self.ocr_engine.ocr(image_bgr, cls=False)
            if result is None:
                return False
            for line in result:
                if line is not None and len(line) > 0:
                    return True
            return False
        else:
            # TrOCR doesn't have a detection mode; try recognition
            text = self.recognize(image_bgr)
            return len(text.strip()) > 2

    def verify_mask(self, original_bgr: np.ndarray, mask: np.ndarray,
                    gt_label: str = None, threshold: float = 0.8) -> tuple:
        """
        Verify mask quality using OCR consistency.

        Args:
            original_bgr: Original text image (BGR)
            mask: Binary mask (0=bg, 255=text)
            gt_label: Optional ground truth label for additional verification
            threshold: Levenshtein ratio threshold for acceptance

        Returns:
            (accepted: bool, ocr_orig: str, ocr_masked: str, score: float)
        """
        # OCR on original image
        ocr_orig = self.recognize(original_bgr)

        # OCR on masked image (text pixels only, rest black)
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
        masked_img = cv2.bitwise_and(original_bgr, mask_3c)
        ocr_masked = self.recognize(masked_img)

        # Compute Levenshtein ratio
        score = self._levenshtein_ratio(ocr_orig, ocr_masked)

        # If GT label is provided, also check against it
        if gt_label:
            gt_score = self._levenshtein_ratio(gt_label, ocr_masked)
            score = max(score, gt_score)

        accepted = score >= threshold
        return accepted, ocr_orig, ocr_masked, score

    @staticmethod
    def _levenshtein_ratio(s1: str, s2: str) -> float:
        """
        Compute normalized Levenshtein similarity ratio.

        Paper formula (Section 4.2):
          ratio = (Len(r_pred) + Len(r_gt) - Dist(r_pred, r_gt))
                / (Len(r_pred) + Len(r_gt))
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        s1 = s1.upper().strip()
        s2 = s2.upper().strip()

        len_s1, len_s2 = len(s1), len(s2)

        # Wagner-Fischer algorithm
        dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
        for i in range(len_s1 + 1):
            dp[i][0] = i
        for j in range(len_s2 + 1):
            dp[0][j] = j

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )

        dist = dp[len_s1][len_s2]
        ratio = (len_s1 + len_s2 - dist) / (len_s1 + len_s2)
        return ratio
