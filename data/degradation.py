"""
Real-ESRGAN style degradation pipeline (two-stage cascaded).

Paper reference:
  Uses the degradation pipeline from Real-ESRGAN [wang2021real], which
  is a TWO-STAGE cascaded degradation model.

  Each stage: Blur → Resize → Noise → JPEG → (optional sinc filter)
  Final: resize to target LR size + optional sinc filter

Reference: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/data/realesrgan_dataset.py

Improvements over previous version:
  - Sinc filter support (critical for Real-ESRGAN fidelity)
  - Multiple blur kernel types (iso/aniso Gaussian, generalized Gaussian, plateau)
  - Poisson noise (shot noise) in addition to Gaussian
  - Per-stage parameter tuning matching the original paper
"""
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


def _generate_sinc_kernel(cutoff: float, kernel_size: int = 21) -> np.ndarray:
    """Generate a sinc filter kernel (band-limiting / ringing artifact).

    This is a key component of Real-ESRGAN degradation that was previously missing.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - half
            y = j - half
            r = np.sqrt(x*x + y*y)
            if r == 0:
                kernel[i, j] = cutoff
            else:
                kernel[i, j] = cutoff * np.sin(np.pi * cutoff * r) / (np.pi * cutoff * r)

    # Normalize
    kernel /= kernel.sum() + 1e-10
    return kernel.astype(np.float32)


def _random_gaussian_kernel(kernel_size: int, sigma_range=(0.2, 3.0)):
    """Generate random isotropic Gaussian blur kernel."""
    sigma = random.uniform(*sigma_range)
    return cv2.getGaussianKernel(kernel_size, sigma) @ \
           cv2.getGaussianKernel(kernel_size, sigma).T


def _random_generalized_gaussian_kernel(kernel_size: int, sigma_range=(0.2, 3.0),
                                         beta_range=(0.5, 4.0)):
    """
    Generalized Gaussian kernel — varies the shape parameter beta.

    Fix D2: Proper implementation using scipy.stats.gennorm instead of
    falling back to standard Gaussian.

    beta < 2: heavier tails (more blur, "plateau"-like)
    beta = 2: standard Gaussian
    beta > 2: lighter tails (sharper)

    Reference: Real-ESRGAN degradation_bsrgan.py
    """
    sigma = random.uniform(*sigma_range)
    beta = random.uniform(*beta_range)

    try:
        from scipy.stats import gennorm
        half = kernel_size // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        # 1D generalized Gaussian
        kernel_1d = gennorm.pdf(x, beta, scale=sigma)
        kernel_1d = kernel_1d / (kernel_1d.sum() + 1e-10)
        # Outer product for 2D kernel
        kernel = np.outer(kernel_1d, kernel_1d)
    except ImportError:
        # Fallback: standard Gaussian if scipy not available
        kernel = cv2.getGaussianKernel(kernel_size, sigma) @ \
                 cv2.getGaussianKernel(kernel_size, sigma).T

    return kernel.astype(np.float32)


def _random_aniso_gaussian_kernel(kernel_size: int, sigma_x_range=(0.2, 3.0),
                                   sigma_y_range=(0.2, 3.0),
                                   rotation_range=(-np.pi, np.pi)):
    """
    Fix D2: Anisotropic Gaussian with rotation support.

    Real-ESRGAN uses rotated anisotropic Gaussian kernels, not just
    axis-aligned different-sigma kernels. The rotation angle produces
    oriented blur artifacts common in real-world degradation.
    """
    sigma_x = random.uniform(*sigma_x_range)
    sigma_y = random.uniform(*sigma_y_range)
    theta = random.uniform(*rotation_range)

    half = kernel_size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    y = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Rotation matrix
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xx_rot = cos_t * xx + sin_t * yy
    yy_rot = -sin_t * xx + cos_t * yy

    kernel = np.exp(-0.5 * (xx_rot**2 / (sigma_x**2 + 1e-10) +
                            yy_rot**2 / (sigma_y**2 + 1e-10)))
    kernel = kernel / (kernel.sum() + 1e-10)
    return kernel.astype(np.float32)


def _random_plateau_kernel(kernel_size: int, sigma_range=(0.2, 3.0),
                            beta_range=(1.0, 2.0)):
    """
    Fix D2: Plateau blur kernel — flat center with Gaussian falloff.

    This is another key kernel type from Real-ESRGAN (biweight kernel).
    The plateau kernel produces uniform blur within a certain radius,
    transitioning smoothly to zero.

    Reference: Real-ESRGAN basicsr/utils/img_process_util.py
    """
    sigma = random.uniform(*sigma_range)
    beta = random.uniform(*beta_range)

    half = kernel_size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    d = np.sqrt(xx**2 + yy**2)

    # Plateau: (1 - (d/sigma)^2)^beta, clipped to >= 0
    kernel = np.maximum(0, 1.0 - (d / (sigma * half + 1e-10))**2) ** beta
    kernel = kernel / (kernel.sum() + 1e-10)
    return kernel.astype(np.float32)


class RealESRGANDegradation:
    """
    Two-stage Real-ESRGAN degradation pipeline with sinc filter support.

    Stage 1: Blur → Resize → Noise → JPEG
    Stage 2: Blur → Resize → Noise → JPEG
    Final:   sinc filter → Resize to target LR

    Fix D2: Now includes all 4 kernel types from Real-ESRGAN:
      1. Isotropic Gaussian
      2. Anisotropic Gaussian (with rotation)
      3. Generalized Gaussian (scipy gennorm)
      4. Plateau (biweight) kernel

    Parameter ranges follow the Real-ESRGAN paper defaults.
    """
    def __init__(self, scale_factor: int = 4, output_size=None, sf=None):
        self.sf = scale_factor if sf is None else sf
        self.output_size = output_size

    def _apply_blur(self, img, sigma_range=(0.2, 3.0)):
        """Random blur with all 4 Real-ESRGAN kernel types.

        Fix D2: Added rotated aniso Gaussian and plateau kernels.
        """
        kernel_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21])

        # Choose kernel type matching Real-ESRGAN probabilities
        kernel_type = random.choices(
            ['iso', 'aniso', 'generalized', 'plateau'],
            weights=[0.45, 0.25, 0.15, 0.15]
        )[0]

        if kernel_type == 'iso':
            kernel = _random_gaussian_kernel(kernel_size, sigma_range)
        elif kernel_type == 'aniso':
            # Fix D2: Use rotated anisotropic Gaussian
            kernel = _random_aniso_gaussian_kernel(kernel_size,
                                                    sigma_x_range=sigma_range,
                                                    sigma_y_range=sigma_range)
        elif kernel_type == 'generalized':
            kernel = _random_generalized_gaussian_kernel(kernel_size, sigma_range)
        else:  # plateau
            kernel = _random_plateau_kernel(kernel_size, sigma_range)

        return cv2.filter2D(img, -1, kernel)

    def _apply_resize(self, img, h, w):
        """Random resize (up/down/keep) with varied interpolation."""
        updown_type = random.choices(["up", "down", "keep"], weights=[0.2, 0.7, 0.1])[0]
        if updown_type == "up":
            scale = random.uniform(1.0, 1.5)
        elif updown_type == "down":
            scale = random.uniform(0.15, 1.0)
        else:
            scale = 1.0

        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)

        interp = random.choice([
            cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4
        ])
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    def _apply_noise(self, img, sigma_range=(1, 30), poisson_prob=0.3):
        """Additive noise: Gaussian or Poisson (shot noise)."""
        img_f = img.astype(np.float32)

        if random.random() < poisson_prob:
            # Poisson noise (more realistic for camera sensors)
            scale = random.uniform(0.05, 3.0)
            vals = len(np.unique(img))
            vals = max(2 ** np.ceil(np.log2(vals)), 1)
            noisy = np.random.poisson(img_f / 255.0 * vals * scale) / (vals * scale) * 255.0
            img_f = noisy.astype(np.float32)
        else:
            # Gaussian noise
            sigma = random.uniform(*sigma_range)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img_f = img_f + noise

        return np.clip(img_f, 0, 255).astype(np.uint8)

    def _apply_jpeg(self, img, quality_range=(30, 95)):
        """JPEG compression artifact."""
        quality = random.randint(*quality_range)
        _, encimg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    def _apply_sinc_filter(self, img, prob=0.5):
        """Apply sinc filter (band-limiting) with given probability.

        Real-ESRGAN uses sinc filter to simulate ringing artifacts caused
        by camera optics and image resizing algorithms.
        """
        if random.random() > prob:
            return img

        # Random cutoff frequency
        cutoff = random.uniform(0.1, 0.9)
        kernel_size = random.choice([7, 9, 11, 13, 15])
        kernel = _generate_sinc_kernel(cutoff, kernel_size)
        return cv2.filter2D(img, -1, kernel)

    def _single_stage(self, img, stage: int = 1):
        """One full degradation pass with stage-specific parameters.

        Stage 1: Wider parameter range
        Stage 2: Slightly tighter (Real-ESRGAN paper convention)
        """
        h, w = img.shape[:2]

        if stage == 1:
            sigma_range = (0.2, 3.0)
            noise_range = (1, 30)
            jpeg_range = (30, 95)
            sinc_prob = 0.1  # Low in first stage
        else:
            sigma_range = (0.2, 1.5)
            noise_range = (1, 25)
            jpeg_range = (30, 95)
            sinc_prob = 0.3  # Higher in second stage

        img = self._apply_blur(img, sigma_range=sigma_range)
        img = self._apply_resize(img, h, w)
        img = self._apply_noise(img, sigma_range=noise_range)
        img = self._apply_jpeg(img, quality_range=jpeg_range)

        return img

    def degrade(self, img):
        """
        Two-stage degradation pipeline.

        Stage 1: Blur → Resize → Noise → JPEG
        Stage 2: Blur → Resize → Noise → JPEG
        Final:   Optional sinc filter → Resize to target LR
        """
        h, w = img.shape[:2]

        if self.output_size is None:
            out_size = (w // self.sf, h // self.sf)
        else:
            out_size = self.output_size

        # Stage 1
        img = self._single_stage(img, stage=1)

        # Stage 2
        img = self._single_stage(img, stage=2)

        # Final sinc filter (Real-ESRGAN applies this before final resize)
        img = self._apply_sinc_filter(img, prob=0.5)

        # Final resize to target LR dimensions
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_CUBIC)

        # Final JPEG compression (optional, Real-ESRGAN does this last)
        if random.random() < 0.5:
            quality = random.randint(30, 95)
            _, encimg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

        return img


def run_degradation_pipeline(hr_dir, lr_dir, sf=4):
    """Batch process HR → LR for an entire directory."""
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    lr_dir.mkdir(parents=True, exist_ok=True)

    degrader = RealESRGANDegradation(scale_factor=sf)

    hr_files = sorted(list(hr_dir.glob("*.png")))
    print(f"Degrading {len(hr_files)} images (two-stage Real-ESRGAN pipeline)...")
    for f in tqdm(hr_files):
        img_hr = cv2.imread(str(f))
        if img_hr is None:
            continue
        img_lr = degrader.degrade(img_hr)
        cv2.imwrite(str(lr_dir / f.name), img_lr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, default="dataset/FTSR/HR")
    parser.add_argument("--lr_dir", type=str, default="dataset/FTSR/LR")
    parser.add_argument("--sf", type=int, default=4)
    args = parser.parse_args()
    run_degradation_pipeline(args.hr_dir, args.lr_dir, args.sf)
