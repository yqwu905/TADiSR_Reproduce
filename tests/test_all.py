#!/usr/bin/env python3
"""
Comprehensive tests for TADiSR model architecture, losses, and data pipeline.

Tests verify:
  1. Model architecture matches paper specifications
  2. Forward pass produces correct output shapes
  3. Loss functions compute correctly
  4. CDIB structure follows paper Fig.2(right)
  5. TACA produces correct a_tex dimensions
  6. Degradation pipeline runs correctly (all 4 kernel types)
  7. Training loop executes without errors
  8. One-step denoising formula is correct
  9. LR image is upsampled before VAE encoding (Fix M4)
  10. Checkpoint save logic correctness (Fix T2)
  11. Focal loss variants (Fix L1)
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_noise_schedule():
    """Test linear beta schedule and denoising formula (Paper Eq.1)."""
    print("=" * 60)
    print("TEST: Noise Schedule & One-step Denoising")
    print("=" * 60)

    from models.tadisr_model import linear_beta_schedule

    betas, alphas, alpha_cumprod = linear_beta_schedule(1000)

    assert betas.shape == (1000,), f"betas shape: {betas.shape}"
    assert alphas.shape == (1000,), f"alphas shape: {alphas.shape}"
    assert alpha_cumprod.shape == (1000,), f"alpha_cumprod shape: {alpha_cumprod.shape}"

    # Verify alpha_cumprod is monotonically decreasing
    assert torch.all(alpha_cumprod[1:] <= alpha_cumprod[:-1]), \
        "alpha_cumprod should be monotonically decreasing"

    # Verify at t=0, almost no noise
    assert alpha_cumprod[0] > 0.99, f"alpha_cumprod[0] = {alpha_cumprod[0]}"

    # Verify at t=200 (fixed timestep), reasonable noise level
    t = 200
    alpha_bar_t = alpha_cumprod[t]
    assert 0.1 < alpha_bar_t < 0.95, f"alpha_bar_200 = {alpha_bar_t}"

    # Test denoising formula: ẑ_H = (1/√ᾱ_t) · z_t - (√(1-ᾱ_t)/√ᾱ_t) · n̂
    coeff1 = 1.0 / torch.sqrt(alpha_bar_t)
    coeff2 = torch.sqrt(1.0 - alpha_bar_t) / torch.sqrt(alpha_bar_t)

    # If noise prediction is perfect (n̂ = noise), we should recover original z
    z_original = torch.randn(1, 4, 8, 8)
    noise = torch.randn_like(z_original)
    z_noisy = torch.sqrt(alpha_bar_t) * z_original + torch.sqrt(1.0 - alpha_bar_t) * noise
    z_recovered = coeff1 * z_noisy - coeff2 * noise

    error = (z_recovered - z_original).abs().max().item()
    assert error < 1e-5, f"Denoising recovery error: {error}"
    print("  ✓ Noise schedule valid")
    print("  ✓ One-step denoising formula correct (recovery error < 1e-5)")
    print(f"  ✓ Fixed timestep t=200: ᾱ_t = {alpha_bar_t:.4f}")
    print()


def test_taca_unet():
    """Test TACA UNet architecture (Paper Section 3.2)."""
    print("=" * 60)
    print("TEST: TACA UNet Architecture")
    print("=" * 60)

    from models.kolors_unet_mod import LightweightTACAUNet

    B, C, H, W = 2, 4, 16, 16
    context_dim = 4096
    seq_len = 77
    num_heads = 4

    unet = LightweightTACAUNet(
        in_channels=C, out_channels=C,
        context_dim=context_dim, base_dim=64,
        num_heads=num_heads, dim_head=32,
    )

    x = torch.randn(B, C, H, W)
    timesteps = torch.full((B,), 200, dtype=torch.long)
    context = torch.randn(B, seq_len, context_dim)
    text_indices = [5]

    noise_pred, a_tex = unet(x, timesteps, context, text_indices)

    # Verify output shapes
    assert noise_pred.shape == (B, C, H, W), \
        f"noise_pred shape: {noise_pred.shape}, expected {(B, C, H, W)}"
    assert a_tex.shape == (B, C, H, W), \
        f"a_tex shape: {a_tex.shape}, expected {(B, C, H, W)}"

    # Verify UNet has skip connections (encoder + decoder)
    assert hasattr(unet, 'down1'), "Missing downsample block 1"
    assert hasattr(unet, 'down2'), "Missing downsample block 2"
    assert hasattr(unet, 'up1'), "Missing upsample block 1"
    assert hasattr(unet, 'up2'), "Missing upsample block 2"

    # Paper: M cross-attention layers
    num_ca = len(unet.ca_layers)
    assert num_ca >= 3, f"Need ≥3 CA layers (encoder+mid+decoder), got {num_ca}"

    # Paper: W_a projection → latent channels
    assert unet.proj_a.in_features == num_ca * num_heads, \
        f"proj_a input: {unet.proj_a.in_features}, expected {num_ca * num_heads}"
    assert unet.proj_a.out_features == C, \
        f"proj_a output: {unet.proj_a.out_features}, expected {C}"

    # Verify all CA layers extracted text attention slices
    for i, ca in enumerate(unet.ca_layers):
        assert ca.a_tex_slice is not None, f"CA layer {i} has no a_tex_slice"

    print(f"  ✓ UNet output shape: noise_pred={noise_pred.shape}, a_tex={a_tex.shape}")
    print(f"  ✓ {num_ca} cross-attention layers with TACA extraction")
    print(f"  ✓ Skip connections: down1→up1, down2→up2")
    print(f"  ✓ W_a projection: {num_ca * num_heads} → {C}")
    print()


def test_cdib():
    """Test CDIB structure (Paper Section 3.3, Fig.2 right)."""
    print("=" * 60)
    print("TEST: Cross-Decoder Interaction Block (CDIB)")
    print("=" * 60)

    from models.vae_jsd import CDIB

    channels = 64
    B, H, W = 2, 16, 16

    cdib = CDIB(channels, num_res_blocks=2, groups=32)
    z_img = torch.randn(B, channels, H, W)
    z_seg = torch.randn(B, channels, H, W)

    out_img, out_seg = cdib(z_img, z_seg)

    # Shape preservation
    assert out_img.shape == z_img.shape, f"img shape: {out_img.shape} vs {z_img.shape}"
    assert out_seg.shape == z_seg.shape, f"seg shape: {out_seg.shape} vs {z_seg.shape}"

    # Paper: zero-initialized output projections → initial output ≈ input
    diff_img = (out_img - z_img).abs().max().item()
    diff_seg = (out_seg - z_seg).abs().max().item()
    assert diff_img < 1e-5, f"With zero-init proj, output should ≈ input. diff={diff_img}"
    assert diff_seg < 1e-5, f"With zero-init proj, output should ≈ input. diff={diff_seg}"

    # Verify structure: ResBlocks + 1×1 split conv → cross-gating → output proj
    assert hasattr(cdib, 'img_res_blocks'), "Missing img_res_blocks"
    assert hasattr(cdib, 'seg_res_blocks'), "Missing seg_res_blocks"
    assert hasattr(cdib, 'img_split_conv'), "Missing img_split_conv"
    assert cdib.img_split_conv.out_channels == channels * 2, \
        "img_split_conv should double channels"

    print(f"  ✓ CDIB preserves shapes: {z_img.shape}")
    print(f"  ✓ Zero-initialized output proj → initial skip-only behavior")
    print(f"  ✓ 2×CDBResBlock → 1×1 Conv(2C) → split → cross-gating")
    print(f"  ✓ GN → SiLU → 1×1 Conv → zero-init residual")
    print()


def test_jsd():
    """Test Joint Segmentation Decoders (Paper Section 3.3)."""
    print("=" * 60)
    print("TEST: Joint Segmentation Decoders (JSD)")
    print("=" * 60)

    from models.vae_jsd import JointSegmentationDecoders, create_jsd

    in_channels = 4
    B, H, W = 2, 8, 8  # Latent spatial dim

    # Use lightweight config for fast testing
    jsd = create_jsd(
        vae=None, latent_channels=in_channels,
        lora_rank=4, lightweight=True,
    )

    z_H = torch.randn(B, in_channels, H, W)
    a_tex = torch.randn(B, in_channels, H, W)

    x_hr, s_mask = jsd(z_H, a_tex)

    # 8× upsampling: latent → pixel space (3 upsamplers)
    expected_h = H * 8
    expected_w = W * 8

    assert x_hr.shape == (B, 3, expected_h, expected_w), \
        f"x_hr shape: {x_hr.shape}, expected {(B, 3, expected_h, expected_w)}"
    assert s_mask.shape == (B, 1, expected_h, expected_w), \
        f"s_mask shape: {s_mask.shape}, expected {(B, 1, expected_h, expected_w)}"

    # Mask should be in [0, 1] (sigmoid output)
    assert s_mask.min() >= 0.0, f"mask min = {s_mask.min()}"
    assert s_mask.max() <= 1.0, f"mask max = {s_mask.max()}"

    # Verify 5 CDIB blocks (mid + 4 up blocks)
    assert len(jsd.cdibs) == 5, f"Should have 5 CDIBs, got {len(jsd.cdibs)}"

    # Verify symmetric decoder structure
    assert hasattr(jsd, 'img_decoder'), "Missing img_decoder"
    assert hasattr(jsd, 'seg_decoder'), "Missing seg_decoder"
    assert len(jsd.img_decoder.up_blocks) == 4, "Should have 4 up blocks"
    assert len(jsd.seg_decoder.up_blocks) == 4, "Should have 4 up blocks"

    # Verify LoRA was applied to Image Decoder
    assert jsd._lora_applied, "LoRA should be applied to Image Decoder"

    # Verify Seg Decoder conv_out outputs 1 channel (mask)
    assert jsd.seg_decoder.conv_out.out_channels == 1, \
        "Seg decoder output should be 1 channel"

    print(f"  ✓ Image decoder: z_H {z_H.shape} → x_hr {x_hr.shape}")
    print(f"  ✓ Seg decoder: a_tex {a_tex.shape} → s_mask {s_mask.shape}")
    print(f"  ✓ 8× spatial upsampling (3 upsamplers)")
    print(f"  ✓ 5 CDIB blocks (mid + 4 up blocks)")
    print(f"  ✓ LoRA applied to Image Decoder")
    print(f"  ✓ Mask output in [0, 1] via sigmoid")
    print()


def test_losses():
    """Test all loss components (Paper Section 3.4)."""
    print("=" * 60)
    print("TEST: Loss Functions")
    print("=" * 60)

    from losses.composite_loss import (
        CompositeTADiSRLoss, ModifiedFocalLoss, DiceLoss,
        AlphaFocalLoss, StandardFocalLoss, SobelOperator
    )

    B = 2
    x_pred = torch.rand(B, 3, 64, 64)
    x_hr = torch.rand(B, 3, 64, 64)
    s_pred = torch.rand(B, 1, 64, 64)
    s_hr = torch.rand(B, 1, 64, 64)

    # Test Sobel operator
    sobel = SobelOperator()
    edge = sobel(s_hr)
    assert edge.shape == s_hr.shape, f"Sobel output shape: {edge.shape}"
    assert edge.min() >= 0, "Sobel magnitude should be >= 0"
    print("  ✓ SobelOperator: single-channel edge detection")

    # Test MFL
    mfl = ModifiedFocalLoss(gamma=2.0)
    loss_mfl = mfl(x_pred, x_hr, s_pred, s_hr)
    assert loss_mfl.ndim == 0, "MFL should return scalar"
    assert loss_mfl.item() >= 0, "MFL should be non-negative"
    print(f"  ✓ ModifiedFocalLoss: {loss_mfl.item():.4f}")

    # Test DiceLoss
    dice = DiceLoss()
    loss_dice = dice(s_pred, s_hr)
    assert 0 <= loss_dice.item() <= 1, f"Dice should be in [0,1], got {loss_dice.item()}"
    print(f"  ✓ DiceLoss: {loss_dice.item():.4f}")

    # Test AlphaFocalLoss
    afocal = AlphaFocalLoss(alpha=0.75, gamma=2.0)
    loss_af = afocal(s_pred, s_hr)
    assert loss_af.item() >= 0, "Alpha focal loss should be non-negative"
    print(f"  ✓ AlphaFocalLoss (α=0.75, γ=2): {loss_af.item():.4f}")

    # Fix L1: Test StandardFocalLoss
    sfocal = StandardFocalLoss(gamma=2.0)
    loss_sf = sfocal(s_pred, s_hr)
    assert loss_sf.item() >= 0, "Standard focal loss should be non-negative"
    print(f"  ✓ StandardFocalLoss (γ=2): {loss_sf.item():.4f}")

    # Test composite loss (default: standard focal, Fix L1)
    criterion = CompositeTADiSRLoss(
        lambda1=5.0, lambda2=10.0, lambda3=10.0, lambda4=1.0
    )
    total, loss_img, loss_seg = criterion(x_pred, x_hr, s_pred, s_hr)
    assert total.item() >= 0, "Total loss should be non-negative"
    assert loss_img.item() >= 0, "Image loss should be non-negative"
    assert loss_seg.item() >= 0, "Seg loss should be non-negative"

    # Verify structure: total = img + seg
    expected_total = loss_img.item() + loss_seg.item()
    diff = abs(total.item() - expected_total)
    assert diff < 1e-3, f"ℓ_tot should = ℓ_img + ℓ_seg. diff={diff}"

    # Test with alpha-balanced focal loss
    criterion_alpha = CompositeTADiSRLoss(use_alpha_focal=True)
    total_a, _, _ = criterion_alpha(x_pred, x_hr, s_pred, s_hr)
    assert total_a.item() >= 0, "Total loss (alpha focal) should be non-negative"
    print(f"  ✓ CompositeLoss (standard focal): total={total.item():.4f}")
    print(f"  ✓ CompositeLoss (alpha focal):    total={total_a.item():.4f}")
    print(f"  ✓ ℓ_tot = ℓ_img + ℓ_seg verified (paper Section 3.4)")
    print()


def test_lr_upsample():
    """Test Fix M4: LR image is upsampled 4× before VAE encoding."""
    print("=" * 60)
    print("TEST: LR Upsample Before VAE (Fix M4)")
    print("=" * 60)

    from models.tadisr_model import TADiSRWrapper

    model = TADiSRWrapper(use_kolors=False, context_dim=4096)

    # Test _upsample_lr
    x_L = torch.rand(2, 3, 32, 32)
    x_L_up = model._upsample_lr(x_L)
    assert x_L_up.shape == (2, 3, 128, 128), \
        f"Expected (2,3,128,128), got {x_L_up.shape}"

    # Verify SR_SCALE is 4
    assert model.SR_SCALE == 4, f"SR_SCALE should be 4, got {model.SR_SCALE}"

    print(f"  ✓ _upsample_lr: {x_L.shape} → {x_L_up.shape} (4× bicubic)")
    print(f"  ✓ SR_SCALE = {model.SR_SCALE}")
    print()


def test_full_model_forward():
    """Test full TADiSR model forward pass (fallback mode)."""
    print("=" * 60)
    print("TEST: Full TADiSR Model Forward Pass")
    print("=" * 60)

    from models.tadisr_model import TADiSRWrapper

    B = 2
    model = TADiSRWrapper(
        use_kolors=False,  # Use lightweight fallback
        context_dim=4096,
    )

    x_L = torch.rand(B, 3, 128, 128)
    context = torch.randn(B, 77, 4096)
    text_indices = [5]

    x_hr_pred, s_mask_pred = model.forward(x_L, context, text_indices)

    # Fix M4: LR is upsampled 4× (128→512) before VAE encoding
    # VAE encodes 512×512 → 64×64 latent (8× downsample)
    # JSD decodes 64×64 → 512×512 (8× upsample)
    assert x_hr_pred.shape[0] == B, f"Batch dim mismatch"
    assert x_hr_pred.shape[1] == 3, f"Image should have 3 channels"
    assert s_mask_pred.shape[1] == 1, f"Mask should have 1 channel"
    assert x_hr_pred.shape[2] == x_hr_pred.shape[3], "Should be square"
    assert s_mask_pred.shape[2] == s_mask_pred.shape[3], "Should be square"

    # Mask in [0, 1]
    assert s_mask_pred.min() >= 0.0 and s_mask_pred.max() <= 1.0, \
        f"Mask range: [{s_mask_pred.min():.3f}, {s_mask_pred.max():.3f}]"

    # Verify fixed timestep
    assert model.FIXED_TIMESTEP == 200, "Paper: t is fixed as 200"
    assert model.TOTAL_TIMESTEPS == 1000, "Standard DDPM: T=1000"

    # forward returns only (x_hr_pred, s_mask_pred), NOT noise/noise_pred
    result = model.forward(x_L, context, text_indices)
    assert len(result) == 2, \
        f"forward should return 2 values (x_hr, s_mask), got {len(result)}"

    # Verify trainable params separation
    trainable = model.get_trainable_params()
    assert len(trainable) > 0, "Should have trainable parameters"

    print(f"  ✓ Input: x_L {x_L.shape}")
    print(f"  ✓ Output: x_hr {x_hr_pred.shape}, s_mask {s_mask_pred.shape}")
    print(f"  ✓ Fixed timestep t=200, T=1000")
    print(f"  ✓ forward returns (x_hr, s_mask) — NO noise prediction output")
    print(f"  ✓ Trainable params: {sum(p.numel() for p in trainable):,}")
    print()


def test_training_loop():
    """Test full training loop (Paper Section 4.1)."""
    print("=" * 60)
    print("TEST: Training Loop (3 steps)")
    print("=" * 60)

    from models.tadisr_model import TADiSRWrapper
    from losses.composite_loss import CompositeTADiSRLoss

    B = 2
    model = TADiSRWrapper(
        use_kolors=False,
        context_dim=4096,
    )

    criterion = CompositeTADiSRLoss()

    # Paper: AdamW, lr=5e-5
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)

    model.train()
    losses = []

    for step in range(3):
        x_L = torch.rand(B, 3, 64, 64)
        x_H = torch.rand(B, 3, 256, 256)
        mask = torch.rand(B, 1, 256, 256)
        context = torch.randn(B, 77, 4096)

        optimizer.zero_grad()

        x_pred, s_pred = model.forward(x_L, context, [5])
        loss, loss_img, loss_seg = criterion(x_pred, x_H, s_pred, mask)

        loss.backward()

        # Gradient clipping (paper doesn't specify, but standard practice)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {step}: loss={loss.item():.4f} "
              f"(img={loss_img.item():.4f}, seg={loss_seg.item():.4f})")

    print(f"  ✓ 3 training steps completed without error")
    print(f"  ✓ Loss = ℓ_img + ℓ_seg (NO noise prediction loss)")
    print(f"  ✓ AdamW optimizer, lr=5e-5")
    print(f"  ✓ Gradient clipping applied (max_norm=1.0)")
    print()


def test_checkpoint_save():
    """Test Fix T2: Checkpoint save logic correctness."""
    print("=" * 60)
    print("TEST: Checkpoint Save Logic (Fix T2)")
    print("=" * 60)

    from models.tadisr_model import TADiSRWrapper

    model = TADiSRWrapper(use_kolors=False, context_dim=4096)
    trainable_params = model.get_trainable_params()

    # Fix T2: Verify that parameter filtering by name works correctly
    trainable_ids = {id(p) for p in trainable_params}
    save_keys = set()
    for name, param in model.named_parameters():
        if id(param) in trainable_ids:
            save_keys.add(name)
        elif 'jsd' in name or 'taca' in name:
            save_keys.add(name)

    model_sd = model.state_dict()
    save_sd = {k: v for k, v in model_sd.items() if k in save_keys}

    assert len(save_sd) > 0, "Should save at least some parameters"

    # Verify JSD params are included
    jsd_keys = [k for k in save_sd if 'jsd' in k]
    assert len(jsd_keys) > 0, "JSD parameters should be in checkpoint"

    # Verify the old broken approach would fail
    # (state_dict tensors are detached copies, requires_grad is always False)
    broken_sd = {k: v for k, v in model_sd.items() if v.requires_grad}
    assert len(broken_sd) == 0, \
        "Old approach (v.requires_grad) should produce empty dict (all False)"

    print(f"  ✓ Name-based filtering: {len(save_sd)} tensors saved")
    print(f"  ✓ JSD params included: {len(jsd_keys)} tensors")
    print(f"  ✓ Old broken approach would save 0 tensors (confirmed bug)")
    print()


def test_degradation_pipeline():
    """Test Real-ESRGAN degradation (Paper Section 3.5)."""
    print("=" * 60)
    print("TEST: Real-ESRGAN Degradation Pipeline")
    print("=" * 60)

    from data.degradation import (
        RealESRGANDegradation, _generate_sinc_kernel,
        _random_generalized_gaussian_kernel, _random_aniso_gaussian_kernel,
        _random_plateau_kernel
    )

    # Test sinc kernel
    kernel = _generate_sinc_kernel(cutoff=0.5, kernel_size=11)
    assert kernel.shape == (11, 11), f"Sinc kernel shape: {kernel.shape}"
    assert abs(kernel.sum() - 1.0) < 0.01, f"Sinc kernel should sum to ~1.0"
    print(f"  ✓ Sinc filter kernel generated: {kernel.shape}")

    # Fix D2: Test generalized Gaussian kernel
    gg_kernel = _random_generalized_gaussian_kernel(11)
    assert gg_kernel.shape == (11, 11), f"GenGauss kernel shape: {gg_kernel.shape}"
    assert abs(gg_kernel.sum() - 1.0) < 0.01, "GenGauss kernel should sum to ~1.0"
    print(f"  ✓ Generalized Gaussian kernel: {gg_kernel.shape}")

    # Fix D2: Test rotated anisotropic Gaussian
    aniso_kernel = _random_aniso_gaussian_kernel(11)
    assert aniso_kernel.shape == (11, 11), f"Aniso kernel shape: {aniso_kernel.shape}"
    assert abs(aniso_kernel.sum() - 1.0) < 0.01, "Aniso kernel should sum to ~1.0"
    print(f"  ✓ Rotated anisotropic Gaussian kernel: {aniso_kernel.shape}")

    # Fix D2: Test plateau kernel
    plateau_kernel = _random_plateau_kernel(11)
    assert plateau_kernel.shape == (11, 11), f"Plateau kernel shape: {plateau_kernel.shape}"
    assert abs(plateau_kernel.sum() - 1.0) < 0.01, "Plateau kernel should sum to ~1.0"
    print(f"  ✓ Plateau (biweight) kernel: {plateau_kernel.shape}")

    # Test degradation
    degrader = RealESRGANDegradation(scale_factor=4)
    img_hr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img_lr = degrader.degrade(img_hr)

    expected_size = (64, 64)
    assert img_lr.shape[:2] == expected_size, \
        f"LR size: {img_lr.shape[:2]}, expected {expected_size}"
    assert img_lr.shape[2] == 3, "Should be 3-channel"
    assert img_lr.dtype == np.uint8, f"Should be uint8, got {img_lr.dtype}"

    # Run multiple times to verify stochastic degradation
    sizes = set()
    for _ in range(5):
        lr = degrader.degrade(img_hr)
        sizes.add(lr.shape[:2])
    assert len(sizes) == 1, "Final LR size should be deterministic"

    print(f"  ✓ Two-stage degradation: 256×256 → {img_lr.shape[:2]}")
    print(f"  ✓ All 4 kernel types implemented (iso, aniso, genGauss, plateau)")
    print(f"  ✓ Sinc filter integrated")
    print(f"  ✓ Consistent output size across stochastic runs")
    print()


def test_dataset():
    """Test dataset loader (Paper Section 4.1)."""
    print("=" * 60)
    print("TEST: Dataset & Collate Function")
    print("=" * 60)

    from data.dataset import TADiSRDataset, tadisr_collate_fn

    # Test context_dim defaults to 4096 (Kolors/ChatGLM)
    ds = TADiSRDataset.__new__(TADiSRDataset)
    assert TADiSRDataset.DEFAULT_CONTEXT_DIM == 4096, \
        f"Default context_dim should be 4096 (Kolors), got {TADiSRDataset.DEFAULT_CONTEXT_DIM}"

    # Test collate function
    batch = [
        {
            "lr": torch.rand(3, 128, 128),
            "hr": torch.rand(3, 512, 512),
            "mask": torch.rand(1, 512, 512),
            "context": torch.rand(77, 4096),
            "text_indices": [5],
            "filename": "test_0.png",
        },
        {
            "lr": torch.rand(3, 128, 128),
            "hr": torch.rand(3, 512, 512),
            "mask": torch.rand(1, 512, 512),
            "context": torch.rand(77, 4096),
            "text_indices": [5],
            "filename": "test_1.png",
        },
    ]

    collated = tadisr_collate_fn(batch)
    assert collated["lr"].shape == (2, 3, 128, 128), "LR batch shape"
    assert collated["hr"].shape == (2, 3, 512, 512), "HR batch shape"
    assert collated["context"].shape == (2, 77, 4096), "Context batch shape"
    assert isinstance(collated["text_indices"], list), "text_indices should be list"
    assert isinstance(collated["filename"], list), "filename should be list"

    print(f"  ✓ Default context_dim = 4096 (Kolors/ChatGLM)")
    print(f"  ✓ Collate: text_indices kept as list (not stacked)")
    print(f"  ✓ Collate: filename kept as list")
    print(f"  ✓ Collate: tensors correctly stacked")
    print()


def test_taca_attn_processor():
    """Test TACAttnProcessor structure (Fix M1)."""
    print("=" * 60)
    print("TEST: TACA AttnProcessor (Fix M1)")
    print("=" * 60)

    from models.tadisr_model import TACAAttnProcessor

    # Verify TACAAttnProcessor exists and has correct interface
    proc = TACAAttnProcessor(text_token_indices=[5], layer_id=0)
    assert proc.text_token_indices == [5], "Should store text indices"
    assert proc.layer_id == 0, "Should store layer id"
    assert proc.a_tex_slice is None, "Should start with None"
    assert callable(proc), "Should be callable"

    print(f"  ✓ TACAAttnProcessor created with text_indices=[5]")
    print(f"  ✓ Callable interface for diffusers processor API")
    print(f"  ✓ a_tex_slice initially None, populated during forward")
    print()


def _reference_cross_attention(attn, hidden_states, encoder_hidden_states, text_indices):
    """Old full-attention math used as a small fp32 correctness oracle."""
    residual = hidden_states
    batch_size = hidden_states.shape[0]
    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    weights = torch.matmul(query, key.transpose(-2, -1)) * (head_dim ** -0.5)
    weights = weights.softmax(dim=-1)
    a_tex = weights[:, :, :, text_indices].mean(dim=-1).contiguous()
    out = torch.matmul(weights, value)
    out = out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    if attn.residual_connection:
        out = out + residual
    out = out / attn.rescale_output_factor
    return out, a_tex


def test_taca_chunked_attention_equivalence_and_grad():
    """Verify chunked TACA slice matches full attention and keeps gradients."""
    print("=" * 60)
    print("TEST: Chunked TACA Attention Equivalence")
    print("=" * 60)

    try:
        from diffusers.models.attention_processor import Attention
    except ImportError:
        print("  ! diffusers not installed, skipping equivalence test")
        print()
        return

    from models.tadisr_model import TACAAttnProcessor

    torch.manual_seed(7)
    attn = Attention(
        query_dim=8,
        cross_attention_dim=8,
        heads=2,
        dim_head=4,
        dropout=0.0,
        bias=True,
        residual_connection=False,
    )
    attn.eval()
    hidden_states = torch.randn(1, 17, 8, requires_grad=True)
    encoder_hidden_states = torch.randn(1, 9, 8, requires_grad=True)
    text_indices = [2, 5]

    ref_out, ref_slice = _reference_cross_attention(
        attn, hidden_states, encoder_hidden_states, text_indices
    )
    proc = TACAAttnProcessor(
        text_token_indices=text_indices,
        layer_id=0,
        query_chunk_size=4,
        use_checkpoint=True,
        detach_taca=False,
    )
    out = proc(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)

    assert torch.allclose(out, ref_out, atol=1e-5), "SDPA output should match full attention"
    assert torch.allclose(proc.a_tex_slice, ref_slice, atol=1e-5), \
        "Chunked TACA slice should match full attention slice"

    loss = out.sum() + proc.a_tex_slice.sum()
    loss.backward()
    assert attn.to_q.weight.grad is not None, "TACA path should keep query gradients"
    assert attn.to_k.weight.grad is not None, "TACA path should keep key gradients"

    print("  ✓ SDPA output matches full attention oracle")
    print("  ✓ Chunked text-token slice matches full attention oracle")
    print("  ✓ Gradients flow through query/key projections")
    print()


def test_lora_failure_is_fatal():
    """LoRA setup failures must not silently train the full UNet."""
    print("=" * 60)
    print("TEST: LoRA Failure Is Fatal")
    print("=" * 60)

    from models.tadisr_model import TADiSRWrapper
    import torch.nn as nn

    model = TADiSRWrapper.__new__(TADiSRWrapper)
    nn.Module.__init__(model)
    model.unet_backbone = nn.Linear(4, 4)
    model.fail_on_lora_error = True

    try:
        model._apply_lora(rank=4)
    except RuntimeError as exc:
        assert "LoRA setup failed" in str(exc), str(exc)
    else:
        raise AssertionError("LoRA failure should raise RuntimeError")

    print("  ✓ LoRA setup failure raises instead of falling back to full-UNet training")
    print()


def test_jsd_dim_controls_channels():
    """Verify jsd_dim is wired into JSD channel width."""
    print("=" * 60)
    print("TEST: jsd_dim Controls JSD Width")
    print("=" * 60)

    from models.vae_jsd import create_jsd

    small = create_jsd(vae=None, latent_channels=4, lora_rank=4, base_channels=16)
    default = create_jsd(vae=None, latent_channels=4, lora_rank=4, lightweight=True)

    assert tuple(small.img_decoder.block_out_channels) == (16, 32, 64, 64), \
        f"Unexpected jsd_dim channels: {small.img_decoder.block_out_channels}"
    small_params = sum(p.numel() for p in small.parameters())
    default_params = sum(p.numel() for p in default.parameters())
    assert small_params < default_params, "Smaller jsd_dim should reduce parameter count"

    print(f"  ✓ jsd_dim=16 → {small.img_decoder.block_out_channels}")
    print(f"  ✓ Parameter count decreases: {small_params:,} < {default_params:,}")
    print()


def test_diffusers_taca_integration():
    """Verify TACA installs correctly on a real diffusers UNet."""
    print("=" * 60)
    print("TEST: Diffusers TACA Integration")
    print("=" * 60)

    try:
        from diffusers import UNet2DConditionModel
    except ImportError:
        print("  ! diffusers not installed, skipping integration test")
        print()
        return

    from models.tadisr_model import TACAAttnProcessor, TACAManager

    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=16,
        norm_num_groups=8,
        attention_head_dim=8,
    )
    manager = TACAManager(unet, text_token_indices=[1], latent_channels=4)

    attn2_keys = [k for k in unet.attn_processors if "attn2.processor" in k]
    assert len(attn2_keys) > 0, "Expected at least one cross-attention processor"
    assert all(isinstance(unet.attn_processors[k], TACAAttnProcessor) for k in attn2_keys), \
        "attn2 processors should be replaced with TACAAttnProcessor"

    x = torch.randn(1, 4, 32, 32)
    t = torch.tensor([10], dtype=torch.long)
    context = torch.randn(1, 5, 16)
    _ = unet(x, t, encoder_hidden_states=context).sample

    assert any(proc.a_tex_slice is not None for proc in manager.processors), \
        "At least one TACA processor should capture attention slices"

    a_tex = manager.aggregate_attention(x.shape)
    spatial_var = a_tex.view(1, 4, -1).var(dim=-1)
    assert torch.any(spatial_var > 0), "Aggregated a_tex should vary spatially"

    state_keys = set(manager.state_dict().keys())
    assert "proj_a.weight" in state_keys and "proj_a.bias" in state_keys, \
        "TACA projection should be part of manager state_dict"

    print(f"  ✓ Replaced {len(attn2_keys)} attn2 processors in diffusers UNet")
    print(f"  ✓ Cross-attention slices captured during forward")
    print(f"  ✓ a_tex has spatial variation (not collapsed to a constant map)")
    print(f"  ✓ proj_a is registered in state_dict")
    print()


def test_vae_normalization_helpers():
    """Verify VAE input/output scaling helpers match diffusers conventions."""
    print("=" * 60)
    print("TEST: VAE Normalization Helpers")
    print("=" * 60)

    from models.tadisr_model import TADiSRWrapper

    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).view(1, 1, 1, 3)
    x_norm = TADiSRWrapper._normalize_vae_input(x)
    expected_norm = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).view(1, 1, 1, 3)
    assert torch.allclose(x_norm, expected_norm), "VAE input should map [0,1] to [-1,1]"

    x_roundtrip = TADiSRWrapper._denormalize_vae_output(x_norm)
    assert torch.allclose(x_roundtrip, x), "VAE output helper should invert normalization"

    x_out = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32).view(1, 1, 1, 3)
    x_denorm = TADiSRWrapper._denormalize_vae_output(x_out)
    expected_denorm = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).view(1, 1, 1, 3)
    assert torch.allclose(x_denorm, expected_denorm), "Output helper should clamp to [0,1]"

    print(f"  ✓ [0,1] → [-1,1] normalization matches diffusers VAE preprocessing")
    print(f"  ✓ [-1,1] → [0,1] denormalization is invertible on-range")
    print(f"  ✓ Out-of-range decoder outputs are clamped back to [0,1]")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TADiSR COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        ("Noise Schedule & Denoising", test_noise_schedule),
        ("TACA UNet Architecture", test_taca_unet),
        ("CDIB Structure", test_cdib),
        ("JSD Architecture", test_jsd),
        ("Loss Functions", test_losses),
        ("LR Upsample (Fix M4)", test_lr_upsample),
        ("TACA AttnProcessor (Fix M1)", test_taca_attn_processor),
        ("Chunked TACA Attention Equivalence", test_taca_chunked_attention_equivalence_and_grad),
        ("LoRA Failure Is Fatal", test_lora_failure_is_fatal),
        ("jsd_dim Controls JSD Width", test_jsd_dim_controls_channels),
        ("Diffusers TACA Integration", test_diffusers_taca_integration),
        ("VAE Normalization Helpers", test_vae_normalization_helpers),
        ("Full Model Forward", test_full_model_forward),
        ("Training Loop", test_training_loop),
        ("Checkpoint Save (Fix T2)", test_checkpoint_save),
        ("Degradation Pipeline", test_degradation_pipeline),
        ("Dataset & Collate", test_dataset),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            import traceback
            print(f"\n  ✗ FAILED: {name}")
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  ✗ {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
