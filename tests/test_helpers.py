"""Test-only lightweight substitutes for production Kolors components."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tadisr_model import linear_beta_schedule
from models.vae_jsd import create_jsd


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
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


class TestTextAwareCrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(0.0))
        self.a_tex_slice = None

    def forward(self, x, context, text_token_indices=None):
        batch, tokens, _ = x.shape
        heads = self.heads
        q = self.to_q(x).view(batch, tokens, heads, self.dim_head).transpose(1, 2)
        k = self.to_k(context).view(batch, -1, heads, self.dim_head).transpose(1, 2)
        v = self.to_v(context).view(batch, -1, heads, self.dim_head).transpose(1, 2)
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * (self.dim_head ** -0.5)
        attn = sim.softmax(dim=-1)
        if text_token_indices:
            a_sub = attn[:, :, :, text_token_indices].mean(dim=-1)
            self.a_tex_slice = a_sub.permute(0, 2, 1)
        else:
            self.a_tex_slice = None
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(batch, tokens, -1)
        return self.to_out(out)


class TestResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class TestDownsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class TestTACAUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        context_dim: int = 4096,
        base_dim: int = 64,
        num_heads: int = 4,
        dim_head: int = 32,
    ):
        super().__init__()
        self.base_dim = base_dim
        d2 = base_dim * 2
        t_emb_dim = base_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.down1_res1 = TestResBlock(base_dim, base_dim, t_emb_dim)
        self.down1_ca = TestTextAwareCrossAttention(base_dim, context_dim, num_heads, dim_head)
        self.down1_res2 = TestResBlock(base_dim, base_dim, t_emb_dim)
        self.down1 = TestDownsample(base_dim)
        self.down2_res1 = TestResBlock(base_dim, d2, t_emb_dim)
        self.down2_ca = TestTextAwareCrossAttention(d2, context_dim, num_heads, dim_head)
        self.down2_res2 = TestResBlock(d2, d2, t_emb_dim)
        self.down2 = TestDownsample(d2)
        self.mid_res1 = TestResBlock(d2, d2, t_emb_dim)
        self.mid_ca = TestTextAwareCrossAttention(d2, context_dim, num_heads, dim_head)
        self.mid_res2 = TestResBlock(d2, d2, t_emb_dim)
        self.up2 = TestUpsample(d2)
        self.up2_res1 = TestResBlock(d2 * 2, d2, t_emb_dim)
        self.up2_ca = TestTextAwareCrossAttention(d2, context_dim, num_heads, dim_head)
        self.up2_res2 = TestResBlock(d2, d2, t_emb_dim)
        self.up1 = TestUpsample(d2)
        self.up1_res1 = TestResBlock(d2 + base_dim, base_dim, t_emb_dim)
        self.up1_ca = TestTextAwareCrossAttention(base_dim, context_dim, num_heads, dim_head)
        self.up1_res2 = TestResBlock(base_dim, base_dim, t_emb_dim)
        self.norm_out = nn.GroupNorm(min(32, base_dim), base_dim)
        self.conv_out = nn.Conv2d(base_dim, out_channels, 3, padding=1)
        self.ca_layers = nn.ModuleList([
            self.down1_ca,
            self.down2_ca,
            self.mid_ca,
            self.up2_ca,
            self.up1_ca,
        ])
        self.proj_a = nn.Linear(len(self.ca_layers) * num_heads, out_channels)

    def _apply_ca(self, x_4d, ca_layer, context, text_token_indices):
        batch, channels, height, width = x_4d.shape
        x_flat = x_4d.view(batch, channels, -1).transpose(1, 2)
        x_flat = ca_layer(x_flat, context, text_token_indices)
        return x_flat.transpose(1, 2).view(batch, channels, height, width)

    def forward(self, x, timesteps, context, text_token_indices=None):
        t_emb = self.time_embed(timestep_embedding(timesteps, self.base_dim))
        h = self.conv_in(x)
        h = self.down1_res1(h, t_emb)
        h = self._apply_ca(h, self.down1_ca, context, text_token_indices) + h
        h = self.down1_res2(h, t_emb)
        skip1 = h
        h = self.down1(h)
        h = self.down2_res1(h, t_emb)
        h = self._apply_ca(h, self.down2_ca, context, text_token_indices) + h
        h = self.down2_res2(h, t_emb)
        skip2 = h
        h = self.down2(h)
        h = self.mid_res1(h, t_emb)
        h = self._apply_ca(h, self.mid_ca, context, text_token_indices) + h
        h = self.mid_res2(h, t_emb)
        h = self.up2(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.up2_res1(h, t_emb)
        h = self._apply_ca(h, self.up2_ca, context, text_token_indices) + h
        h = self.up2_res2(h, t_emb)
        h = self.up1(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.up1_res1(h, t_emb)
        h = self._apply_ca(h, self.up1_ca, context, text_token_indices) + h
        h = self.up1_res2(h, t_emb)
        noise_pred = self.conv_out(F.silu(self.norm_out(h)))
        a_tex = self.aggregate_text_attention(x.shape)
        return noise_pred, a_tex

    def aggregate_text_attention(self, target_shape):
        slices = []
        batch, _, height, width = target_shape
        device = next(self.parameters()).device
        for ca in self.ca_layers:
            if ca.a_tex_slice is None:
                continue
            a = ca.a_tex_slice
            spatial = a.shape[1]
            size = int(math.sqrt(spatial))
            if size * size == spatial:
                a = a.permute(0, 2, 1).view(batch, -1, size, size)
                a = F.interpolate(a, size=(height, width), mode="bilinear", align_corners=False)
                a = a.flatten(2).permute(0, 2, 1)
            slices.append(a)
        if not slices:
            return torch.zeros(batch, self.proj_a.out_features, height, width, device=device)
        a_cat = torch.cat(slices, dim=-1)
        a_proj = self.proj_a(a_cat)
        return a_proj.permute(0, 2, 1).view(batch, -1, height, width)


class TestVAEEncoder(nn.Module):
    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)


class TestUNet(nn.Module):
    def __init__(self, latent_channels: int = 4, context_dim: int = 4096):
        super().__init__()
        self.unet = TestTACAUNet(
            in_channels=latent_channels,
            out_channels=latent_channels,
            context_dim=context_dim,
            base_dim=32,
        )

    def forward(self, x, timesteps, context, text_token_indices=None):
        return self.unet(x, timesteps, context, text_token_indices)


class TestTADiSRWrapper(nn.Module):
    SR_SCALE = 4
    FIXED_TIMESTEP = 200
    TOTAL_TIMESTEPS = 1000

    def __init__(
        self,
        context_dim: int = 4096,
        latent_channels: int = 4,
        lora_rank: int = 2,
        jsd_dim: Optional[int] = 16,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.latent_channels = latent_channels
        self.use_real_backbone = False
        self.vae_encoder = TestVAEEncoder(latent_channels)
        self.unet = TestUNet(latent_channels, context_dim)
        self.jsd = create_jsd(
            vae=None,
            latent_channels=latent_channels,
            lora_rank=lora_rank,
            base_channels=jsd_dim,
            layers_per_block=1,
        )
        betas, alphas, alpha_cumprod = linear_beta_schedule(self.TOTAL_TIMESTEPS)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

    def _upsample_lr(self, x_l):
        return F.interpolate(x_l, scale_factor=self.SR_SCALE, mode="bicubic", align_corners=False)

    def _get_schedule_params(self, t: int, device):
        alpha_bar_t = self.alpha_cumprod[t]
        coeff1 = 1.0 / torch.sqrt(alpha_bar_t)
        coeff2 = torch.sqrt(1.0 - alpha_bar_t) / torch.sqrt(alpha_bar_t)
        return coeff1.to(device), coeff2.to(device)

    def _add_noise(self, z, t: int):
        alpha_bar_t = self.alpha_cumprod[t]
        noise = torch.randn_like(z)
        z_noisy = torch.sqrt(alpha_bar_t) * z + torch.sqrt(1.0 - alpha_bar_t) * noise
        return z_noisy, noise

    def forward(self, x_l, context=None, text_token_indices=None, return_debug: bool = False):
        batch = x_l.shape[0]
        device = x_l.device
        if context is None:
            context = torch.randn(batch, 77, self.context_dim, device=device)
        if text_token_indices is None:
            text_token_indices = [5]
        z_l = self.vae_encoder(self._upsample_lr(x_l))
        z_l_noisy, _ = self._add_noise(z_l, self.FIXED_TIMESTEP)
        timesteps = torch.full((batch,), self.FIXED_TIMESTEP, dtype=torch.long, device=device)
        noise_pred, a_tex = self.unet(z_l_noisy, timesteps, context, text_token_indices)
        coeff1, coeff2 = self._get_schedule_params(self.FIXED_TIMESTEP, device)
        z_h = coeff1 * z_l_noisy - coeff2 * noise_pred
        x_hr_pred, s_mask_pred = self.jsd(z_h, a_tex)
        if return_debug:
            return x_hr_pred, s_mask_pred, {"a_tex": a_tex}
        return x_hr_pred, s_mask_pred

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_trainable_param_summary(self):
        return {"test_model": sum(p.numel() for p in self.get_trainable_params())}

    def assert_trainable_parameter_contract(self):
        return None
