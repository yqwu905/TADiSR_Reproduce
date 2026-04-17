"""
Lightweight UNet with Text-Aware Cross-Attention (TACA) for CPU testing.

This module provides a structurally-faithful but lightweight UNet that can
run on CPU for testing. When Kolors backbone is available, this is NOT used —
the real Kolors UNet2DConditionModel is used instead.

Paper reference (Section 3.2):
  - Cross-attention maps from ALL M layers responding to the "text" token
    are extracted, CONCATENATED along the token channel, and linearly
    projected to match the latent dimension of z_H.

Architecture improvements over original:
  - Proper encoder with spatial downsampling (2 levels)
  - Proper decoder with spatial upsampling (2 levels)
  - Skip connections between encoder and decoder
  - Timestep conditioning via sinusoidal embedding + MLP
  - Multi-layer cross-attention with TACA
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Timestep embedding (standard sinusoidal, same as LDM / DDPM)
# ---------------------------------------------------------------------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Create sinusoidal timestep embeddings (Vaswani et al.)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# Single Text-Aware Cross-Attention layer
# ---------------------------------------------------------------------------
class TextAwareCrossAttention(nn.Module):
    """
    Cross-attention layer that additionally extracts the attention
    response to the "text" token index(es) and stores it for later
    aggregation across all M layers.
    """
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0),
        )
        self.a_tex_slice = None

    def forward(self, x, context, text_token_indices=None):
        B, N, _ = x.shape
        h = self.heads

        q = self.to_q(x).view(B, N, h, self.dim_head).transpose(1, 2)
        k = self.to_k(context).view(B, -1, h, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(B, -1, h, self.dim_head).transpose(1, 2)

        scale = self.dim_head ** -0.5
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * scale
        attn = sim.softmax(dim=-1)

        # TACA: extract attention slice for "text" token(s)
        if text_token_indices is not None and len(text_token_indices) > 0:
            a_sub = attn[:, :, :, text_token_indices].mean(dim=-1)
            self.a_tex_slice = a_sub.permute(0, 2, 1)  # [B, N, H]
        else:
            self.a_tex_slice = None

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# ResBlock with timestep conditioning and optional channel change
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# Downsample / Upsample blocks
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ---------------------------------------------------------------------------
# Lightweight TACA UNet with proper U-Net architecture
# ---------------------------------------------------------------------------
class LightweightTACAUNet(nn.Module):
    """
    Structurally faithful lightweight UNet with TACA for CPU testing.

    Architecture:
      Encoder:  conv_in → [ResBlock, CA, ResBlock, Down] × 2
      Mid:      ResBlock, CA, ResBlock
      Decoder:  [Up, ResBlock+skip, CA, ResBlock] × 2 → conv_out
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 context_dim: int = 4096, base_dim: int = 64,
                 num_heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.base_dim = base_dim
        d2 = base_dim * 2  # Second level channels
        t_emb_dim = base_dim * 4

        # --- Timestep MLP ---
        self.time_embed = nn.Sequential(
            nn.Linear(base_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        # --- Encoder ---
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # Level 1: base_dim
        self.down1_res1 = ResBlock(base_dim, base_dim, t_emb_dim)
        self.down1_ca = TextAwareCrossAttention(base_dim, context_dim, num_heads, dim_head)
        self.down1_res2 = ResBlock(base_dim, base_dim, t_emb_dim)
        self.down1 = Downsample(base_dim)

        # Level 2: d2
        self.down2_res1 = ResBlock(base_dim, d2, t_emb_dim)
        self.down2_ca = TextAwareCrossAttention(d2, context_dim, num_heads, dim_head)
        self.down2_res2 = ResBlock(d2, d2, t_emb_dim)
        self.down2 = Downsample(d2)

        # --- Mid ---
        self.mid_res1 = ResBlock(d2, d2, t_emb_dim)
        self.mid_ca = TextAwareCrossAttention(d2, context_dim, num_heads, dim_head)
        self.mid_res2 = ResBlock(d2, d2, t_emb_dim)

        # --- Decoder ---
        # Level 2: d2 (with skip from encoder d2)
        self.up2 = Upsample(d2)
        self.up2_res1 = ResBlock(d2 * 2, d2, t_emb_dim)  # concat skip
        self.up2_ca = TextAwareCrossAttention(d2, context_dim, num_heads, dim_head)
        self.up2_res2 = ResBlock(d2, d2, t_emb_dim)

        # Level 1: base_dim (with skip from encoder base_dim)
        self.up1 = Upsample(d2)
        self.up1_res1 = ResBlock(d2 + base_dim, base_dim, t_emb_dim)  # concat skip
        self.up1_ca = TextAwareCrossAttention(base_dim, context_dim, num_heads, dim_head)
        self.up1_res2 = ResBlock(base_dim, base_dim, t_emb_dim)

        # --- Output ---
        self.norm_out = nn.GroupNorm(min(32, base_dim), base_dim)
        self.conv_out = nn.Conv2d(base_dim, out_channels, 3, padding=1)

        # --- All CA layers (for TACA aggregation) ---
        self.ca_layers = nn.ModuleList([
            self.down1_ca, self.down2_ca,
            self.mid_ca,
            self.up2_ca, self.up1_ca,
        ])
        self.num_ca_layers = len(self.ca_layers)

        # --- Paper Eq.4: W_a projection ---
        self.proj_a = nn.Linear(self.num_ca_layers * num_heads, out_channels)

    def _apply_ca(self, x_4d, ca_layer, context, text_token_indices):
        """Apply cross-attention to a 4D feature map."""
        B, C, H, W = x_4d.shape
        x_flat = x_4d.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        x_flat = ca_layer(x_flat, context, text_token_indices)
        return x_flat.transpose(1, 2).view(B, C, H, W)

    def forward(self, x, timesteps, context, text_token_indices=None):
        """
        Returns:
            noise_pred: [B, C, H, W]
            a_tex:      [B, C_out, H, W] projected text attention map
        """
        # Timestep embedding
        t_emb = timestep_embedding(timesteps, self.base_dim)
        t_emb = self.time_embed(t_emb)

        # --- Encoder ---
        h = self.conv_in(x)

        # Level 1
        h = self.down1_res1(h, t_emb)
        h = self._apply_ca(h, self.down1_ca, context, text_token_indices) + h
        h = self.down1_res2(h, t_emb)
        skip1 = h  # Save for skip connection
        h = self.down1(h)

        # Level 2
        h = self.down2_res1(h, t_emb)
        h = self._apply_ca(h, self.down2_ca, context, text_token_indices) + h
        h = self.down2_res2(h, t_emb)
        skip2 = h
        h = self.down2(h)

        # --- Mid ---
        h = self.mid_res1(h, t_emb)
        h = self._apply_ca(h, self.mid_ca, context, text_token_indices) + h
        h = self.mid_res2(h, t_emb)

        # --- Decoder ---
        # Level 2
        h = self.up2(h)
        h = torch.cat([h, skip2], dim=1)  # Skip connection
        h = self.up2_res1(h, t_emb)
        h = self._apply_ca(h, self.up2_ca, context, text_token_indices) + h
        h = self.up2_res2(h, t_emb)

        # Level 1
        h = self.up1(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.up1_res1(h, t_emb)
        h = self._apply_ca(h, self.up1_ca, context, text_token_indices) + h
        h = self.up1_res2(h, t_emb)

        # Output
        noise_pred = self.conv_out(F.silu(self.norm_out(h)))

        # --- Aggregate text-attention from ALL M layers ---
        a_tex = self._aggregate_text_attention(x.shape)
        return noise_pred, a_tex

    def _aggregate_text_attention(self, input_shape):
        """
        Paper Eq.3-4:
          a_tex_raw = Concat(Search(a^1, c_tex), ..., Search(a^M, c_tex))
          a_tex = W_a · a_tex_raw

        Fix M5: Use proper 2D spatial interpolation instead of 1D bilinear
        on a fake [N', 1] extent.
        """
        B, C, H, W = input_shape
        N = H * W

        slices = []
        for ca in self.ca_layers:
            if ca.a_tex_slice is not None:
                s = ca.a_tex_slice  # [B, N', heads]
                N_layer = s.shape[1]
                num_heads = s.shape[2]

                if N_layer != N:
                    # Fix M5: Reshape to 2D spatial grid for proper interpolation
                    import math as _math
                    h_layer = int(_math.sqrt(N_layer))
                    w_layer = N_layer // h_layer if h_layer > 0 else 1
                    # Find best spatial factors
                    if h_layer * w_layer != N_layer:
                        for h_try in range(int(_math.sqrt(N_layer)), 0, -1):
                            if N_layer % h_try == 0:
                                h_layer = h_try
                                w_layer = N_layer // h_try
                                break

                    # [B, N', heads] → [B, heads, h, w]
                    s_4d = s.permute(0, 2, 1).view(B, num_heads, h_layer, w_layer)
                    # Proper 2D bilinear interpolation
                    s_4d = F.interpolate(s_4d, size=(H, W), mode="bilinear",
                                         align_corners=False)
                    # [B, heads, H, W] → [B, N, heads]
                    s = s_4d.view(B, num_heads, N).permute(0, 2, 1)
                slices.append(s)
            else:
                slices.append(torch.zeros(
                    B, N, self.ca_layers[0].heads,
                    device=next(self.parameters()).device
                ))

        # Concat → [B, N, M*heads], then project → [B, N, out_channels]
        a_concat = torch.cat(slices, dim=-1)
        a_tex = self.proj_a(a_concat)
        a_tex = a_tex.transpose(1, 2).view(B, -1, H, W)
        return a_tex
