"""
Joint Segmentation Decoders (JSD) with Cross-Decoder Interaction Blocks (CDIB).

Paper reference (Section 3.3):
  - Image Decoder (D^v_ϕ): mirrors VAE decoder structure, LoRA fine-tuning
  - Text Segmentation Decoder (D^s_φ): symmetric, randomly initialized
  - CDIB at each resolution level for cross-decoder feature interaction

Architecture faithfully mirrors Kolors VAE decoder (AutoencoderKL):
  block_out_channels = [128, 256, 512, 512]
  layers_per_block   = 2 → num_resnets_per_block = 3 (layers_per_block + 1)
  latent_channels    = 4
  norm_num_groups    = 32
  act_fn             = silu

Decoder structure (reversed channels [512, 512, 256, 128]):
  conv_in:       Conv2d(4, 512)
  mid_block:     ResNet(512) → Attention(512) → ResNet(512)
  up_block_0:    3×ResNet(512→512) + Upsample    (2× spatial)
  up_block_1:    3×ResNet(512→512) + Upsample    (2× spatial)
  up_block_2:    3×ResNet(512→256) + Upsample    (2× spatial)
  up_block_3:    3×ResNet(256→128)               (no upsample)
  conv_norm_out: GroupNorm(32, 128) → SiLU
  conv_out:      Conv2d(128, 3)  [Image] or Conv2d(128, 1)  [Seg]

Total spatial upscale: 8× (3 upsamplers)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA Wrappers
# ---------------------------------------------------------------------------
class LoRAConv2d(nn.Module):
    """Conv2d with frozen base weights + trainable low-rank adapter.

    output = base(x) + lora_B(lora_A(x)) * scale
    """
    def __init__(self, base_conv: nn.Conv2d, rank: int = 16, scale: float = 1.0):
        super().__init__()
        self.base = base_conv
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        c_in = base_conv.in_channels
        c_out = base_conv.out_channels
        self.lora_A = nn.Conv2d(c_in, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, c_out, 1, bias=False)
        self.scale = scale

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


# ---------------------------------------------------------------------------
# Building Blocks — exact diffusers naming for weight loading compatibility
# ---------------------------------------------------------------------------
class ResnetBlock2D(nn.Module):
    """Matches diffusers ResnetBlock2D naming and forward order.

    Forward: norm1 → act → conv1 → norm2 → act → conv2 + shortcut
    State dict keys: norm1, conv1, norm2, conv2, conv_shortcut (optional)
    """
    def __init__(self, in_channels, out_channels, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = x
        h = self.nonlinearity(self.norm1(x))
        h = self.conv1(h)
        h = self.nonlinearity(self.norm2(h))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + h


class SelfAttention(nn.Module):
    """Self-attention for VAE mid block.

    Matches diffusers Attention naming: group_norm, to_q, to_k, to_v, to_out
    """
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(num_groups, channels, eps=1e-6)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        h = self.group_norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]

        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(C)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        out = self.to_out[0](out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out + residual


class Upsample2D(nn.Module):
    """Nearest-neighbor 2× upsample + Conv2d. Matches diffusers naming."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UpDecoderBlock2D(nn.Module):
    """Matches diffusers UpDecoderBlock2D: N resnets + optional upsample.

    State dict keys: resnets.{i}.*, upsamplers.0.conv.*
    """
    def __init__(self, in_channels, out_channels, num_layers=3,
                 add_upsample=True, num_groups=32):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_ch = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(input_ch, out_channels, num_groups))
        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels)])

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x)
        return x

    def forward_resnets_only(self, x):
        """Run only resnets (used by JSD to insert CDIB before upsample)."""
        for resnet in self.resnets:
            x = resnet(x)
        return x

    def forward_upsample_only(self, x):
        """Run only upsampler (used by JSD after CDIB)."""
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x)
        return x


class UNetMidBlock2D(nn.Module):
    """VAE decoder mid block: ResNet → Attention → ResNet.

    State dict keys: resnets.{i}.*, attentions.0.*
    """
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, num_groups),
            ResnetBlock2D(channels, channels, num_groups),
        ])
        self.attentions = nn.ModuleList([
            SelfAttention(channels, num_groups),
        ])

    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


# ---------------------------------------------------------------------------
# CDIB — Cross-Decoder Interaction Block (Section 3.3, Fig. 2)
# ---------------------------------------------------------------------------
class CDBResBlock(nn.Module):
    """Simple ResBlock used inside CDIB (not from VAE, randomly initialized)."""
    def __init__(self, channels, groups=32):
        super().__init__()
        # Clamp groups to not exceed channels
        groups = min(groups, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class CDIB(nn.Module):
    """Cross-Decoder Interaction Block.

    Paper Eq. 4-5:
      f^v_out, f^s_out = CDIB(f^v_in, f^s_in)

    Each branch:
      1. 2× CDBResBlock to refine features
      2. 1×1 conv → split into (gate, value) along channel dim
      3. Cross-gating: output_v = σ(gate_s) ⊙ value_v
      4. GN → SiLU → 1×1 conv to project back + zero-init skip
    """
    def __init__(self, channels, num_res_blocks=2, groups=32):
        super().__init__()
        groups = min(groups, channels)

        # Image branch processing
        self.img_res_blocks = nn.ModuleList(
            [CDBResBlock(channels, groups) for _ in range(num_res_blocks)]
        )
        # Seg branch processing
        self.seg_res_blocks = nn.ModuleList(
            [CDBResBlock(channels, groups) for _ in range(num_res_blocks)]
        )

        # Split convs: project to 2× channels, split into (gate, value)
        self.img_split_conv = nn.Conv2d(channels, channels * 2, 1)
        self.seg_split_conv = nn.Conv2d(channels, channels * 2, 1)

        # Output projection (zero-init for stable training start)
        self.img_out_norm = nn.GroupNorm(groups, channels)
        self.img_out_proj = nn.Conv2d(channels, channels, 1)
        self.seg_out_norm = nn.GroupNorm(groups, channels)
        self.seg_out_proj = nn.Conv2d(channels, channels, 1)
        self.act = nn.SiLU()

        # Zero-initialize output projections for residual stability
        nn.init.zeros_(self.img_out_proj.weight)
        nn.init.zeros_(self.img_out_proj.bias)
        nn.init.zeros_(self.seg_out_proj.weight)
        nn.init.zeros_(self.seg_out_proj.bias)

    def forward(self, f_img, f_seg):
        # Refine features
        h_img = f_img
        for blk in self.img_res_blocks:
            h_img = blk(h_img)
        h_seg = f_seg
        for blk in self.seg_res_blocks:
            h_seg = blk(h_seg)

        # Split into gate and value
        img_gv = self.img_split_conv(h_img)
        seg_gv = self.seg_split_conv(h_seg)
        img_gate, img_val = img_gv.chunk(2, dim=1)
        seg_gate, seg_val = seg_gv.chunk(2, dim=1)

        # Cross-gating: image uses seg's gate and vice versa
        out_img = torch.sigmoid(seg_gate) * img_val
        out_seg = torch.sigmoid(img_gate) * seg_val

        # Output projection with zero-init residual
        out_img = self.img_out_proj(self.act(self.img_out_norm(out_img)))
        out_seg = self.seg_out_proj(self.act(self.seg_out_norm(out_seg)))

        return f_img + out_img, f_seg + out_seg


# ---------------------------------------------------------------------------
# Decoder Branch — single decoder stream
# ---------------------------------------------------------------------------
class DecoderBranch(nn.Module):
    """One decoder branch matching Kolors VAE decoder architecture.

    This module uses exact diffusers attribute naming so VAE weights can be
    loaded via state_dict key matching.

    Args:
        in_channels:  Latent channel count (4 for Kolors)
        out_channels: Output channels (3 for image, 1 for segmentation)
        block_out_channels: Per-block channel counts [128, 256, 512, 512]
        layers_per_block: ResNets per block (diffusers convention, actual = +1)
        norm_num_groups: GroupNorm groups
    """
    def __init__(self, in_channels=4, out_channels=3,
                 block_out_channels=(128, 256, 512, 512),
                 layers_per_block=2, norm_num_groups=32):
        super().__init__()
        self.block_out_channels = block_out_channels

        reversed_channels = list(reversed(block_out_channels))
        # reversed_channels = [512, 512, 256, 128] for default config

        # --- conv_in ---
        self.conv_in = nn.Conv2d(in_channels, reversed_channels[0], 3, padding=1)

        # --- mid_block ---
        self.mid_block = UNetMidBlock2D(reversed_channels[0], norm_num_groups)

        # --- up_blocks ---
        num_layers = layers_per_block + 1  # diffusers convention
        up_blocks = []
        output_channel = reversed_channels[0]
        for i, ch in enumerate(reversed_channels):
            prev_output_channel = output_channel
            output_channel = ch
            is_final = (i == len(reversed_channels) - 1)
            up_blocks.append(UpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=num_layers,
                add_upsample=not is_final,
                num_groups=norm_num_groups,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # --- conv_norm_out + conv_out ---
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, reversed_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(reversed_channels[-1], out_channels, 3, padding=1)

    def forward(self, z):
        """Full single-stream forward (without CDIB). Used for standalone testing."""
        h = self.conv_in(z)
        h = self.mid_block(h)
        for block in self.up_blocks:
            h = block(h)
        h = self.conv_act(self.conv_norm_out(h))
        h = self.conv_out(h)
        return h

    def get_cdib_channels(self):
        """Return channel dims at each CDIB insertion point.

        CDIBs are placed:
          [0] after mid_block
          [1] after up_block_0 resnets (before upsample)
          [2] after up_block_1 resnets (before upsample)
          [3] after up_block_2 resnets (before upsample)
          [4] after up_block_3 resnets
        """
        rev = list(reversed(self.block_out_channels))
        return [rev[0]] + rev  # [512, 512, 512, 256, 128] for default


# ---------------------------------------------------------------------------
# Joint Segmentation Decoders — Main Module
# ---------------------------------------------------------------------------
class JointSegmentationDecoders(nn.Module):
    """TADiSR Joint Segmentation Decoders (JSD).

    Paper Section 3.3:
      Image Decoder D^v_ϕ: mirrors VAE decoder, LoRA fine-tuned
      Seg Decoder D^s_φ: symmetric structure, randomly initialized
      CDIB at each resolution level for cross-decoder interaction

    Args:
        block_out_channels: VAE decoder config [128, 256, 512, 512]
        layers_per_block:   ResNets per block in diffusers convention (2)
        latent_channels:    Input latent channels (4)
        lora_rank:          LoRA adapter rank for image decoder
        norm_num_groups:    GroupNorm groups (32)
    """
    def __init__(self, block_out_channels=(128, 256, 512, 512),
                 layers_per_block=2, latent_channels=4,
                 lora_rank=16, norm_num_groups=32):
        super().__init__()
        self.lora_rank = lora_rank

        # Image decoder: output 3 channels (RGB image)
        self.img_decoder = DecoderBranch(
            in_channels=latent_channels,
            out_channels=3,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        # Segmentation decoder: output 1 channel (binary mask)
        self.seg_decoder = DecoderBranch(
            in_channels=latent_channels,
            out_channels=1,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        # CDIBs at each resolution level
        cdib_channels = self.img_decoder.get_cdib_channels()
        self.cdibs = nn.ModuleList([
            CDIB(ch, num_res_blocks=2, groups=min(norm_num_groups, ch))
            for ch in cdib_channels
        ])

        self._lora_applied = False
        self._vae_loaded = False

    # -------------------------------------------------------------------
    # Weight loading from Kolors VAE
    # -------------------------------------------------------------------
    def init_from_vae(self, vae):
        """Load VAE decoder weights into Image Decoder.

        Matches state dict keys by name since our DecoderBranch uses
        identical attribute naming as diffusers' Decoder.
        """
        vae_sd = vae.decoder.state_dict()
        img_sd = self.img_decoder.state_dict()

        loaded_keys = []
        skipped_keys = []
        for key in img_sd:
            if key in vae_sd:
                if img_sd[key].shape == vae_sd[key].shape:
                    img_sd[key] = vae_sd[key]
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(
                        f"  {key}: ours={img_sd[key].shape} vs vae={vae_sd[key].shape}"
                    )
            else:
                skipped_keys.append(f"  {key}: not in VAE state dict")

        self.img_decoder.load_state_dict(img_sd)
        self._vae_loaded = True

        print(f"[JSD] Loaded {len(loaded_keys)}/{len(img_sd)} tensors from VAE decoder")
        if skipped_keys:
            print(f"[JSD] Skipped {len(skipped_keys)} keys:")
            for s in skipped_keys[:10]:
                print(s)
            if len(skipped_keys) > 10:
                print(f"  ... and {len(skipped_keys) - 10} more")

        return len(loaded_keys), len(skipped_keys)

    # -------------------------------------------------------------------
    # LoRA application for Image Decoder
    # -------------------------------------------------------------------
    def apply_lora(self, rank=None):
        """Replace Conv2d layers in Image Decoder with LoRA-wrapped versions.

        Paper: Image Decoder uses LoRA fine-tuned parameters ϕ.
        Base weights are frozen; only LoRA adapters are trainable.
        GroupNorm parameters are also frozen.
        """
        if self._lora_applied:
            print("[JSD] LoRA already applied, skipping.")
            return

        rank = rank or self.lora_rank
        count = 0

        for name, module in list(self.img_decoder.named_modules()):
            if isinstance(module, nn.Conv2d):
                # Navigate to parent and replace
                parts = name.split('.')
                parent = self.img_decoder
                for part in parts[:-1]:
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)
                attr = parts[-1]

                lora_conv = LoRAConv2d(module, rank=rank)
                if attr.isdigit():
                    parent[int(attr)] = lora_conv
                else:
                    setattr(parent, attr, lora_conv)
                count += 1

        # Freeze all non-LoRA parameters in Image Decoder
        for name, param in self.img_decoder.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad = False

        self._lora_applied = True
        print(f"[JSD] Applied LoRA (rank={rank}) to {count} Conv2d layers in Image Decoder")
        self._print_param_summary()

    def _print_param_summary(self):
        """Print trainable vs frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[JSD] Parameters: total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
        print(f"[JSD]   Image Decoder (LoRA):"
              f" {sum(p.numel() for p in self.img_decoder.parameters() if p.requires_grad):,} trainable")
        print(f"[JSD]   Seg Decoder:"
              f" {sum(p.numel() for p in self.seg_decoder.parameters()):,} total (all trainable)")
        print(f"[JSD]   CDIBs:"
              f" {sum(p.numel() for p in self.cdibs.parameters()):,} total (all trainable)")

    # -------------------------------------------------------------------
    # Forward pass — interleaved with CDIBs
    # -------------------------------------------------------------------
    def forward(self, z_H, a_tex):
        """JSD forward: parallel Image + Seg decoders with CDIB interaction.

        Args:
            z_H:   Denoised latent [B, 4, H/8, W/8]
            a_tex: TACA attention map projected to latent dim [B, 4, H/8, W/8]

        Returns:
            x_hr:   Reconstructed HR image [B, 3, H, W]
            s_mask: Text segmentation mask [B, 1, H, W] (sigmoid-activated)
        """
        # --- conv_in ---
        h_img = self.img_decoder.conv_in(z_H)
        h_seg = self.seg_decoder.conv_in(a_tex)

        # --- mid_block ---
        h_img = self.img_decoder.mid_block(h_img)
        h_seg = self.seg_decoder.mid_block(h_seg)
        h_img, h_seg = self.cdibs[0](h_img, h_seg)

        # --- up_blocks with CDIB after resnets, before upsample ---
        for i, (img_block, seg_block) in enumerate(
            zip(self.img_decoder.up_blocks, self.seg_decoder.up_blocks)
        ):
            # Process resnets
            h_img = img_block.forward_resnets_only(h_img)
            h_seg = seg_block.forward_resnets_only(h_seg)

            # CDIB interaction (index offset by 1 because cdib[0] is mid)
            h_img, h_seg = self.cdibs[i + 1](h_img, h_seg)

            # Upsample (if not final block)
            h_img = img_block.forward_upsample_only(h_img)
            h_seg = seg_block.forward_upsample_only(h_seg)

        # --- Output heads ---
        h_img = self.img_decoder.conv_act(self.img_decoder.conv_norm_out(h_img))
        x_hr = self.img_decoder.conv_out(h_img)

        h_seg = self.seg_decoder.conv_act(self.seg_decoder.conv_norm_out(h_seg))
        s_mask = torch.sigmoid(self.seg_decoder.conv_out(h_seg))

        return x_hr, s_mask


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def create_jsd(vae=None, block_out_channels=None, layers_per_block=None,
               latent_channels=4, lora_rank=16, lightweight=False):
    """Create JSD, optionally loading from a Kolors VAE.

    Args:
        vae: Optional AutoencoderKL instance. If provided, reads config from
             it and loads decoder weights into Image Decoder.
        block_out_channels: Override block_out_channels. If None, reads from
                            VAE config or uses default.
        layers_per_block: Override. Default from VAE or 2.
        latent_channels: Latent channel count. Default 4.
        lora_rank: LoRA adapter rank. Default 16.
        lightweight: If True and no VAE, use smaller [32, 64, 128, 128] config.
    """
    if vae is not None:
        # Read architecture from VAE config
        cfg = vae.config
        boc = block_out_channels or list(cfg.get("block_out_channels", [128, 256, 512, 512]))
        lpb = layers_per_block if layers_per_block is not None else cfg.get("layers_per_block", 2)
        lch = cfg.get("latent_channels", latent_channels)
        nng = cfg.get("norm_num_groups", 32)
    else:
        if lightweight:
            boc = block_out_channels or [32, 64, 128, 128]
            lpb = layers_per_block if layers_per_block is not None else 1
        else:
            boc = block_out_channels or [128, 256, 512, 512]
            lpb = layers_per_block if layers_per_block is not None else 2
        lch = latent_channels
        nng = 32

    # Ensure GroupNorm groups doesn't exceed smallest channel count
    nng = min(nng, min(boc))

    jsd = JointSegmentationDecoders(
        block_out_channels=tuple(boc),
        layers_per_block=lpb,
        latent_channels=lch,
        lora_rank=lora_rank,
        norm_num_groups=nng,
    )

    if vae is not None:
        jsd.init_from_vae(vae)

    # Apply LoRA to Image Decoder
    jsd.apply_lora(rank=lora_rank)

    return jsd
