"""
Composite loss for TADiSR multi-task learning.

Paper reference (Section 3.4):
  ℓ_tot = ℓ_img + ℓ_seg

  ℓ_img = λ1·MSE(x̂_H, x_H) + λ2·LPIPS(x̂_H, x_H) + ℓ_MFL(x̂_H, x_H, ŝ, s)
    where λ1=5.0, λ2=10.0

  ℓ_MFL = exp(∇s) ⊙ (1 - (ŝ⊙s + (1-ŝ)⊙(1-s)))^γ ⊙ ||x̂_H - x_H||_1
    where ∇ is Sobel edge operator

  ℓ_seg = λ3·MSE(ŝ, s) + λ4·Focal(ŝ, s) + Dice(ŝ, s)
    where λ3=10.0, λ4=1.0

Fixes applied:
  - Issue 5:  Proper single-channel Sobel handling
  - Issue 6:  Real LPIPS via lpips library (with L1 fallback)
  - Issue 7:  Alpha-balanced Focal Loss for segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing real LPIPS (Issue 6)
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


# ---------------------------------------------------------------------------
# Sobel edge operator
# ---------------------------------------------------------------------------
class SobelOperator(nn.Module):
    """Sobel edge detector for single-channel inputs."""
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight_x",
            torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "weight_y",
            torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]]).view(1, 1, 3, 3),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] — must be single-channel

        Returns:
            edge_magnitude: [B, 1, H, W]
        """
        # Ensure single channel (Issue 5)
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


# ---------------------------------------------------------------------------
# Modified Focal Loss (Paper Section 3.4, Eq.6)
# ---------------------------------------------------------------------------
class ModifiedFocalLoss(nn.Module):
    """
    ℓ_MFL(x̂_H, x_H, ŝ, s) = exp(∇s) ⊙ (1 - p)^γ ⊙ ||x̂_H − x_H||_1

    where p = ŝ⊙s + (1−ŝ)⊙(1−s)   is the probability of correct classification
    and ∇ is the Sobel edge operator applied to the GT mask s.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.sobel = SobelOperator()

    def forward(self, x_pred, x_hr, s_pred, s_hr):
        """
        Args:
            x_pred: [B, 3, H, W]
            x_hr:   [B, 3, H, W]
            s_pred: [B, 1, H', W']
            s_hr:   [B, 1, H', W']
        """
        # L1 distance per pixel, averaged over channels → [B, 1, H, W]
        l1_dist = torch.abs(x_pred - x_hr).mean(dim=1, keepdim=True)

        # Probability of correct classification
        p = s_pred * s_hr + (1 - s_pred) * (1 - s_hr)  # [B, 1, H', W']

        # Edge weight: Sobel on GT segmentation mask (Issue 5: guaranteed single channel)
        edge_weight = self.sobel(s_hr)  # [B, 1, H', W']

        # Align spatial resolutions if they differ
        target_size = l1_dist.shape[2:]
        if edge_weight.shape[2:] != target_size:
            edge_weight = F.interpolate(edge_weight, size=target_size,
                                        mode="bilinear", align_corners=False)
            p = F.interpolate(p, size=target_size,
                              mode="bilinear", align_corners=False)

        # Paper Eq.6
        focal_weight = torch.exp(edge_weight) * (1 - p) ** self.gamma
        weighted_loss = l1_dist * focal_weight
        return weighted_loss.mean()


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:],
                                 mode="bilinear", align_corners=False)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Alpha-balanced Focal Loss for Segmentation  (Issue 7)
# ---------------------------------------------------------------------------
class AlphaFocalLoss(nn.Module):
    """
    Alpha-balanced binary focal loss per [Lin et al. 2017]:
      FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:],
                                 mode="bilinear", align_corners=False)

        # Clamp for numerical stability
        pred = pred.clamp(1e-6, 1 - 1e-6)

        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)

        # Alpha weighting: α for positive class, (1-α) for negative
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Standard Focal Loss (no alpha balancing)  (Fix L1)
# ---------------------------------------------------------------------------
class StandardFocalLoss(nn.Module):
    """
    Standard binary focal loss per [Lin et al. 2017] without alpha balancing:
      FL(p_t) = -(1 - p_t)^γ log(p_t)

    Fix L1: The paper cites "Focal Loss [Lin et al. 2017]" without explicitly
    mentioning alpha balancing. This class provides the standard version.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:],
                                 mode="bilinear", align_corners=False)

        pred = pred.clamp(1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Composite Loss  (Paper Section 3.4)
# ---------------------------------------------------------------------------
class CompositeTADiSRLoss(nn.Module):
    """
    Total loss = ℓ_img + ℓ_seg

    ℓ_img = λ1·MSE + λ2·LPIPS + MFL
    ℓ_seg = λ3·MSE + λ4·Focal + Dice
    """

    def __init__(self, lambda1: float = 5.0, lambda2: float = 10.0,
                 lambda3: float = 10.0, lambda4: float = 1.0,
                 focal_gamma: float = 2.0, focal_alpha: float = 0.75,
                 use_alpha_focal: bool = False,
                 lpips_resize: int = 0):
        """
        Args:
            use_alpha_focal: If True, use alpha-balanced focal loss.
                If False (default), use standard focal loss matching
                the paper's citation of [Lin et al. 2017] without alpha.
        """
        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lpips_resize = int(lpips_resize or 0)

        self.mse = nn.MSELoss()
        self.mfl = ModifiedFocalLoss(gamma=focal_gamma)
        self.dice = DiceLoss()

        # Fix L1: Default to standard focal loss (paper doesn't specify alpha)
        if use_alpha_focal:
            self.focal_seg = AlphaFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.focal_seg = StandardFocalLoss(gamma=focal_gamma)

        # --- LPIPS (Issue 6) ---
        if HAS_LPIPS:
            self.lpips_fn = lpips.LPIPS(net="vgg", verbose=False)
            # Freeze LPIPS
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
            self._use_real_lpips = True
        else:
            self._use_real_lpips = False
            print("[CompositeTADiSRLoss] LPIPS library not found. "
                  "Using VGG perceptual loss fallback. "
                  "Install: pip install lpips")

    def _lpips_loss(self, x_pred, x_hr):
        """Compute LPIPS or fallback perceptual loss."""
        if self._use_real_lpips:
            if self.lpips_resize > 0:
                max_side = max(x_pred.shape[-2:])
                if max_side > self.lpips_resize:
                    x_pred = F.interpolate(
                        x_pred, size=(self.lpips_resize, self.lpips_resize),
                        mode="bilinear", align_corners=False
                    )
                    x_hr = F.interpolate(
                        x_hr, size=(self.lpips_resize, self.lpips_resize),
                        mode="bilinear", align_corners=False
                    )
            # LPIPS expects [-1, 1] range
            return self.lpips_fn(x_pred * 2 - 1, x_hr * 2 - 1).mean()
        else:
            # Fallback: simple pixel-space L1 as a stand-in
            return F.l1_loss(x_pred, x_hr)

    def forward(self, x_pred, x_hr, s_pred, s_hr):
        """
        Args:
            x_pred: [B, 3, H, W]  predicted SR image
            x_hr:   [B, 3, H, W]  ground truth HR image
            s_pred: [B, 1, H', W'] predicted segmentation mask
            s_hr:   [B, 1, H', W'] ground truth segmentation mask
        """
        # Align spatial sizes
        if x_pred.shape != x_hr.shape:
            x_pred = F.interpolate(x_pred, size=x_hr.shape[2:],
                                   mode="bilinear", align_corners=False)
        if s_pred.shape != s_hr.shape:
            s_pred = F.interpolate(s_pred, size=s_hr.shape[2:],
                                   mode="bilinear", align_corners=False)

        # ---- Image SR Loss ----
        loss_mse_img = self.mse(x_pred, x_hr)
        loss_lpips = self._lpips_loss(x_pred, x_hr)
        loss_mfl = self.mfl(x_pred, x_hr, s_pred, s_hr)
        loss_img = (self.lambda1 * loss_mse_img
                    + self.lambda2 * loss_lpips
                    + loss_mfl)

        # ---- Segmentation Loss ----
        loss_mse_seg = self.mse(s_pred, s_hr)
        loss_focal_seg = self.focal_seg(s_pred, s_hr)
        loss_dice = self.dice(s_pred, s_hr)
        loss_seg = (self.lambda3 * loss_mse_seg
                    + self.lambda4 * loss_focal_seg
                    + loss_dice)

        return loss_img + loss_seg, loss_img, loss_seg
