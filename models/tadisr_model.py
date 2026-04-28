"""
TADiSR model wrapper — integrates Kolors backbone (VAE + UNet + ChatGLM Text Encoder)
with Text-Aware Cross-Attention (TACA) and Joint Segmentation Decoders (JSD).

Paper reference (Section 3.1, 4.1):
  - Built upon the Kolors variant of LDMs
  - VAE Encoder E_θ encodes x_L → z_L
  - Fixed prompt "A high-quality photo with clear text"
  - ChatGLM text encoder → c_y, extract c_tex for "text" token
  - UNet predicts noise n̂ with LoRA fine-tuning on cross-attention
  - One-step denoising at fixed t=200: ẑ_H = α_t · z_{L,t} − β_t · n̂
  - JSD: (ẑ_H, a_tex) → (x̂_H, ŝ)

  Total loss: ℓ_tot = ℓ_img + ℓ_seg  (NO separate noise prediction loss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional, List

from .vae_jsd import JointSegmentationDecoders, create_jsd


# ---------------------------------------------------------------------------
# Linear beta schedule (standard DDPM, T=1000)
# ---------------------------------------------------------------------------
def linear_beta_schedule(timesteps: int = 1000, beta_start: float = 0.00085,
                         beta_end: float = 0.012):
    """Returns betas, alphas, alpha_cumprod for a linear schedule."""
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return betas.float(), alphas.float(), alpha_cumprod.float()


# ---------------------------------------------------------------------------
# Try to load real Kolors components from diffusers
# ---------------------------------------------------------------------------
def _try_load_kolors_vae(pretrained_path: str = "Kwai-Kolors/Kolors-diffusers"):
    """Load Kolors VAE (AutoencoderKL) from HuggingFace."""
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            pretrained_path, subfolder="vae",
            torch_dtype=torch.float32,
        )
        vae.requires_grad_(False)  # Freeze VAE encoder
        print(f"[TADiSR] ✓ Loaded Kolors VAE from {pretrained_path}")
        return vae, vae.config.latent_channels
    except Exception as e:
        print(f"[TADiSR] ✗ Could not load Kolors VAE: {e}")
        return None, 4


def _try_load_kolors_unet(pretrained_path: str = "Kwai-Kolors/Kolors-diffusers"):
    """Load Kolors UNet2DConditionModel from HuggingFace."""
    try:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_path, subfolder="unet",
            torch_dtype=torch.float32,
        )
        print(f"[TADiSR] ✓ Loaded Kolors UNet from {pretrained_path}")
        return unet
    except Exception as e:
        print(f"[TADiSR] ✗ Could not load Kolors UNet: {e}")
        return None


def _try_load_kolors_text_encoder(pretrained_path: str = "Kwai-Kolors/Kolors-diffusers"):
    """Load ChatGLM text encoder used by Kolors."""
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path, subfolder="text_encoder", trust_remote_code=True
        )
        text_encoder = AutoModel.from_pretrained(
            pretrained_path, subfolder="text_encoder", trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        print(f"[TADiSR] ✓ Loaded ChatGLM text encoder from {pretrained_path}")
        return tokenizer, text_encoder
    except Exception as e:
        print(f"[TADiSR] ✗ Could not load ChatGLM text encoder: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Lightweight fallback components for CPU testing
# ---------------------------------------------------------------------------
class _FallbackVAEEncoder(nn.Module):
    """Lightweight VAE encoder stub for CPU testing."""
    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, latent_channels * 2, 3, padding=1),  # mean + logvar
        )
        self.latent_channels = latent_channels

    def forward(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        # Reparameterize (or just use mean for deterministic encoding)
        return mean  # [B, latent_channels, H/8, W/8]


class _FallbackVAEDecoder(nn.Module):
    """Lightweight VAE decoder stub for CPU testing."""
    def __init__(self, latent_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, z):
        return self.decoder(z)


class _FallbackUNet(nn.Module):
    """
    Lightweight UNet with proper architecture for CPU testing.
    Includes: down/up sampling, skip connections, timestep conditioning,
    and multi-layer cross-attention with text-attention extraction.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 context_dim: int = 4096, base_dim: int = 64):
        super().__init__()
        from .kolors_unet_mod import LightweightTACAUNet
        self.unet = LightweightTACAUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            context_dim=context_dim,
            base_dim=base_dim,
        )

    def forward(self, x, timesteps, context, text_token_indices=None):
        return self.unet(x, timesteps, context, text_token_indices)


# ---------------------------------------------------------------------------
# TACA Custom Attention Processor for diffusers UNet  (Fix M1)
# ---------------------------------------------------------------------------
class TACAAttnProcessor(nn.Module):
    """
    Custom attention processor that captures cross-attention maps for
    the "text" token, implementing the TACA mechanism for diffusers UNet.

    Paper (Section 3.2):
      a_tex_raw = Concat(Search(a^1, c_tex), ..., Search(a^M, c_tex))
      a_tex = W_a · a_tex_raw

    This processor replaces the default AttnProcessor2_0 in cross-attention
    layers (attn2) of the UNet. It computes standard cross-attention output
    AND stores the attention weights for the "text" token indices.
    """

    def __init__(self, text_token_indices: List[int], layer_id: int):
        super().__init__()
        self.text_token_indices = text_token_indices
        self.layer_id = layer_id
        # Will be populated during forward pass
        self.a_tex_slice = None  # [B, num_heads, spatial_dim]

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        """
        Standard cross-attention forward with TACA attention extraction.

        Args:
            attn: The Attention module
            hidden_states: [B, N, C] spatial features (query source)
            encoder_hidden_states: [B, S, C] text embeddings (key/value source)
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if encoder_hidden_states is None:
            # Self-attention — don't capture, use standard path
            encoder_hidden_states = hidden_states

        if attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        # Project Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Compute attention weights explicitly (cannot use F.scaled_dot_product_attention
        # because we need the attention weight matrix)
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        # attn_weights: [B, heads, N, S]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.softmax(dim=-1)

        # TACA: Extract attention response for "text" token indices
        if self.text_token_indices:
            # attn_weights[:, :, :, text_indices] -> [B, heads, N, len(indices)]
            # Average across the text token indices, keep per-head
            a_sub = attn_weights[:, :, :, self.text_token_indices]  # [B, H, N, T]
            a_sub = a_sub.mean(dim=-1)  # [B, H, N] — average over text tokens
            self.a_tex_slice = a_sub.contiguous()  # Keep gradients for joint training

        # Standard attention output
        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection out
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class TACAManager(nn.Module):
    """
    Manages TACA attention extraction across all cross-attention layers
    of a diffusers UNet.

    Replaces attn2 processors with TACAAttnProcessor instances,
    then aggregates their captured attention maps after each forward pass.
    """

    def __init__(self, unet, text_token_indices: List[int], latent_channels: int = 4):
        super().__init__()
        self.text_token_indices = text_token_indices
        self.latent_channels = latent_channels
        self.processors: List[TACAAttnProcessor] = []
        self._num_heads_list: List[int] = []

        # Install TACA processors on all cross-attention (attn2) layers
        self._install_processors(unet)

        # W_a projection: [M * heads] → latent_channels
        total_heads = sum(self._num_heads_list)
        self.proj_a = nn.Linear(total_heads, latent_channels)
        print(f"[TACA] Installed {len(self.processors)} TACA processors, "
              f"total_heads={total_heads}, proj → {latent_channels}")

    def _install_processors(self, unet):
        """Replace attn2 (cross-attention) processors with TACAAttnProcessor."""
        attn_procs = {}
        direct_replacements = []
        layer_id = 0

        for name, module in unet.named_modules():
            if hasattr(module, 'to_q') and hasattr(module, 'to_k'):
                if 'attn2' in name:
                    num_heads = getattr(module, 'heads', 8)
                    proc = TACAAttnProcessor(
                        text_token_indices=self.text_token_indices,
                        layer_id=layer_id,
                    )
                    self.processors.append(proc)
                    self._num_heads_list.append(num_heads)

                    # diffusers expects the full "<attention_path>.processor" key.
                    proc_name = f"{name}.processor"
                    attn_procs[proc_name] = proc
                    direct_replacements.append((module, proc))
                    layer_id += 1

        if not attn_procs:
            return

        installed = 0
        try:
            current_procs = dict(unet.attn_processors)
            for key, proc in attn_procs.items():
                if key in current_procs:
                    current_procs[key] = proc
                    installed += 1
            if installed:
                unet.set_attn_processor(current_procs)
        except Exception as e:
            print(f"[TACA] set_attn_processor failed: {e}")
            installed = 0

        if installed != len(attn_procs):
            installed = 0
            for module, proc in direct_replacements:
                if hasattr(module, 'set_processor'):
                    module.set_processor(proc)
                    installed += 1

        print(f"[TACA] Replaced {installed} attn2 processors")

    def aggregate_attention(self, target_shape) -> torch.Tensor:
        """
        Aggregate text-attention from ALL M cross-attention layers.

        Paper Eq.3-4:
          a_tex_raw = Concat(Search(a^1, c_tex), ..., Search(a^M, c_tex))
          a_tex = W_a · a_tex_raw

        Args:
            target_shape: (B, C, H, W) — target spatial dimensions for a_tex

        Returns:
            a_tex: [B, latent_channels, H, W]
        """
        B, C, H, W = target_shape
        N = H * W
        device = self.proj_a.weight.device

        slices = []
        for idx, proc in enumerate(self.processors):
            if proc.a_tex_slice is not None:
                s = proc.a_tex_slice  # [B, heads, N']
                num_heads = s.shape[1]
                N_layer = s.shape[2]

                # Reshape to 2D spatial for proper interpolation
                h_layer = int(math.sqrt(N_layer))
                w_layer = N_layer // h_layer if h_layer > 0 else 1

                if h_layer * w_layer != N_layer:
                    # Non-square: find best factors
                    for h_try in range(int(math.sqrt(N_layer)), 0, -1):
                        if N_layer % h_try == 0:
                            h_layer = h_try
                            w_layer = N_layer // h_try
                            break

                s_4d = s.view(B, num_heads, h_layer, w_layer)  # [B, heads, h, w]

                if h_layer != H or w_layer != W:
                    s_4d = F.interpolate(
                        s_4d.float(), size=(H, W),
                        mode="bilinear", align_corners=False
                    )

                # [B, heads, H, W] → [B, H*W, heads]
                s_flat = s_4d.view(B, num_heads, N).permute(0, 2, 1)
                slices.append(s_flat)
            else:
                # Fallback: zero slice
                default_heads = self._num_heads_list[idx] if idx < len(self._num_heads_list) else 8
                slices.append(torch.zeros(B, N, default_heads, device=device))

        # Concat → [B, N, M*heads], then project → [B, N, latent_channels]
        a_concat = torch.cat(slices, dim=-1).to(device)
        a_tex = self.proj_a(a_concat)  # [B, N, latent_channels]
        a_tex = a_tex.permute(0, 2, 1).view(B, -1, H, W)  # [B, latent_channels, H, W]
        return a_tex

    def clear(self):
        """Clear stored attention maps."""
        for proc in self.processors:
            proc.a_tex_slice = None


# ---------------------------------------------------------------------------
# Main TADiSR Wrapper
# ---------------------------------------------------------------------------
class TADiSRWrapper(nn.Module):
    """
    End-to-end TADiSR wrapper for training.

    Architecture (Paper Section 3.1):
      1. VAE encoder: x_L → (bicubic 4× up) → z_L  (frozen Kolors VAE)
      2. Text encoder: prompt → c_y  (frozen ChatGLM)
      3. Add noise to z_L at fixed t=200 → z_{L,t}
      4. UNet (with LoRA on cross-attention): predicts noise n̂
      5. Extract cross-attention response to "text" token → a_tex
      6. One-step denoising: ẑ_H = α_t · z_{L,t} − β_t · n̂
      7. JSD: (ẑ_H, a_tex) → (x̂_H, ŝ)

    Training loss:
      ℓ_tot = ℓ_img + ℓ_seg  (NO separate noise loss)
    """

    FIXED_TIMESTEP = 200   # Paper: "t is fixed as 200"
    TOTAL_TIMESTEPS = 1000
    FIXED_PROMPT = "A high-quality photo with clear text"
    SR_SCALE = 4  # 4× super-resolution

    def __init__(self, pretrained_path: str = "Kwai-Kolors/Kolors-diffusers",
                 use_kolors: bool = True, latent_channels: int = 4,
                 context_dim: int = 4096, lora_rank: int = 16,
                 precomputed_text_context_path: Optional[str] = None):
        super().__init__()
        self.latent_channels = latent_channels
        self.context_dim = context_dim
        self.use_real_backbone = False
        self.precomputed_text_context = None
        self.precomputed_text_indices = None
        self.precomputed_text_context_path = precomputed_text_context_path

        # --- Try loading real Kolors backbone ---
        self.vae = None
        self.unet_backbone = None
        self.tokenizer = None
        self.text_encoder = None
        self._text_token_indices = None
        self.taca_manager = None  # Fix M1: TACA manager for real UNet

        if use_kolors:
            self._try_init_kolors(pretrained_path, lora_rank)

        # --- Fallback to lightweight components for CPU testing ---
        if not self.use_real_backbone:
            print("[TADiSR] Using lightweight fallback mode (CPU testing)")
            self._init_fallback(latent_channels, context_dim)

        # --- JSD (always custom, per paper) ---
        self.jsd = create_jsd(
            vae=self.vae if self.use_real_backbone else None,
            latent_channels=latent_channels,
            lora_rank=lora_rank,
            lightweight=not self.use_real_backbone,
        )

        # --- Noise schedule ---
        betas, alphas, alpha_cumprod = linear_beta_schedule(self.TOTAL_TIMESTEPS)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

    @staticmethod
    def _normalize_vae_input(x: torch.Tensor) -> torch.Tensor:
        """Match diffusers VAE preprocessing: [0, 1] → [-1, 1]."""
        return x.clamp(0.0, 1.0).mul(2.0).sub(1.0)

    @staticmethod
    def _denormalize_vae_output(x: torch.Tensor) -> torch.Tensor:
        """Match diffusers image postprocess: [-1, 1] → [0, 1]."""
        return x.div(2.0).add(0.5).clamp(0.0, 1.0)

    def _try_init_kolors(self, pretrained_path, lora_rank):
        """Initialize with real Kolors backbone + LoRA."""
        # VAE
        self.vae, lc = _try_load_kolors_vae(pretrained_path)
        if self.vae is None:
            return
        self.latent_channels = lc

        # UNet
        self.unet_backbone = _try_load_kolors_unet(pretrained_path)
        if self.unet_backbone is None:
            self.vae = None
            return

        # Text encoder or offline precomputed prompt embedding
        if self.precomputed_text_context_path:
            ok = self._load_precomputed_text_context(self.precomputed_text_context_path)
            if not ok:
                self.vae = None
                self.unet_backbone = None
                return
            self._text_token_indices = self.precomputed_text_indices
            print("[TADiSR] ✓ Using precomputed text context; ChatGLM is not loaded.")
        else:
            self.tokenizer, self.text_encoder = _try_load_kolors_text_encoder(pretrained_path)
            if self.tokenizer is None:
                self.vae = None
                self.unet_backbone = None
                return
            # Tokenize fixed prompt and find "text" token index
            self._text_token_indices = self._find_text_token_indices()

        # Fix M1: Install TACA attention processors BEFORE LoRA
        # (so LoRA wraps the already-installed TACA processors)
        self.taca_manager = TACAManager(
            self.unet_backbone,
            text_token_indices=self._text_token_indices,
            latent_channels=self.latent_channels,
        )

        # Apply LoRA to UNet cross-attention layers
        self._apply_lora(lora_rank)

        # Get context_dim from UNet config
        self.context_dim = self.unet_backbone.config.cross_attention_dim

        self.use_real_backbone = True
        print(f"[TADiSR] ✓ Kolors backbone initialized with LoRA rank={lora_rank}")


    def _normalize_single_prompt_context(self, context: torch.Tensor, source: str) -> torch.Tensor:
        """Normalize context tensor to shape [1, S, C] for a fixed prompt."""
        if context.ndim == 2:
            context = context.unsqueeze(0)

        if context.ndim != 3:
            raise ValueError(
                f"{source}: expected 2D/3D tensor, got shape {tuple(context.shape)}"
            )

        # Some ChatGLM variants return [S, B, C]. For a fixed prompt we expect B=1.
        if context.shape[1] == 1 and context.shape[0] != 1:
            context = context.transpose(0, 1).contiguous()

        if context.shape[0] != 1:
            raise ValueError(
                f"{source}: expected fixed-prompt batch dim=1, got shape {tuple(context.shape)}"
            )
        return context

    def _load_precomputed_text_context(self, path: str) -> bool:
        """Load precomputed fixed-prompt context from disk.

        Supported file formats:
          1) Tensor file containing [S, C] or [1, S, C]
          2) Dict with keys:
             - "context": Tensor [S, C] or [1, S, C]
             - optional "text_token_indices": list[int]
        """
        if not os.path.isfile(path):
            print(f"[TADiSR] ✗ Precomputed context file not found: {path}")
            return False
        try:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                context = obj.get("context")
                indices = obj.get("text_token_indices")
            else:
                context = obj
                indices = None

            if not isinstance(context, torch.Tensor):
                raise ValueError("missing tensor 'context'")

            context = self._normalize_single_prompt_context(context, source="precomputed context")
            self.precomputed_text_context = context.float()
            self.context_dim = int(context.shape[-1])

            if indices is None:
                indices = [context.shape[1] - 1]
                print("[TADiSR] ⚠ text_token_indices missing; defaulting to last token index.")
            self.precomputed_text_indices = [int(i) for i in indices]
            print(
                f"[TADiSR] ✓ Loaded precomputed text context from {path}, "
                f"shape={tuple(self.precomputed_text_context.shape)}, "
                f"text_token_indices={self.precomputed_text_indices}"
            )
            return True
        except Exception as e:
            print(f"[TADiSR] ✗ Failed to load precomputed context: {e}")
            return False

    def _apply_lora(self, rank):
        """Apply LoRA to UNet cross-attention layers (attn2 only) via peft.

        Paper: "LoRA-based fine-tuning strategy... fine-tune the
        cross-attention mechanism between text and image tokens"

        Fix M2: Only target attn2 (cross-attention) modules, NOT
        attn1 (self-attention). Previous code used target_modules=
        ["to_q", "to_k", ...] which matched BOTH attn1 and attn2.
        """
        try:
            from peft import LoraConfig, get_peft_model

            # Collect only attn2 (cross-attention) linear layer names
            cross_attn_modules = []
            for name, module in self.unet_backbone.named_modules():
                if 'attn2' in name and (
                    name.endswith('to_q')
                    or name.endswith('to_k')
                    or name.endswith('to_v')
                    or name.endswith('to_out.0')
                ):
                    cross_attn_modules.append(name)

            if not cross_attn_modules:
                # Fallback: if direct name matching fails (e.g. diffusers
                # version difference), use regex-based targeting
                print("[TADiSR] Direct attn2 name matching found 0 modules, "
                      "using regex pattern fallback")
                import re
                lora_config = LoraConfig(
                    r=rank,
                    lora_alpha=rank,
                    target_modules=re.compile(r".*attn2\..*(?:to_q|to_k|to_v|to_out\.0)"),
                    lora_dropout=0.0,
                )
            else:
                lora_config = LoraConfig(
                    r=rank,
                    lora_alpha=rank,
                    target_modules=cross_attn_modules,
                    lora_dropout=0.0,
                )

            self.unet_backbone = get_peft_model(self.unet_backbone, lora_config)

            # Report
            trainable = sum(p.numel() for p in self.unet_backbone.parameters()
                            if p.requires_grad)
            total = sum(p.numel() for p in self.unet_backbone.parameters())
            print(f"[TADiSR] LoRA (cross-attn only): {trainable:,} / {total:,} "
                  f"params trainable ({100*trainable/total:.2f}%)")
            print(f"[TADiSR] Targeted {len(cross_attn_modules)} attn2 modules")
        except Exception as e:
            print(f"[TADiSR] LoRA setup failed: {e}. UNet fully trainable as fallback.")

    def _find_text_token_indices(self) -> List[int]:
        """Tokenize the fixed prompt and find positions of 'text' token."""
        if self.tokenizer is None:
            return [10]  # Fallback guess
        tokens = self.tokenizer.tokenize(self.FIXED_PROMPT)
        indices = [i for i, t in enumerate(tokens) if 'text' in t.lower()]
        if not indices:
            indices = [len(tokens) - 1]  # Last token as fallback
        print(f"[TADiSR] Prompt tokens: {tokens}")
        print(f"[TADiSR] 'text' token indices: {indices}")
        return indices

    def _init_fallback(self, latent_channels, context_dim):
        """Initialize lightweight fallback for CPU testing."""
        self.vae_encoder_fb = _FallbackVAEEncoder(latent_channels)
        self.unet_fb = _FallbackUNet(
            in_channels=latent_channels,
            out_channels=latent_channels,
            context_dim=context_dim,
        )
        self._text_token_indices = [5]  # Placeholder
        # Generate a fixed random context for testing
        torch.manual_seed(42)
        self._fallback_context = nn.Parameter(
            torch.randn(1, 77, context_dim), requires_grad=False
        )

    def _upsample_lr(self, x_L):
        """
        Fix M4: Upsample LR image by SR_SCALE (4×) before VAE encoding.

        Paper: 4× super-resolution. The LR image must be upsampled to HR
        resolution before VAE encoding, as the VAE expects HR-sized inputs.
        This follows the approach of OSEDiff [wu2025one] and SupIR [yu2024scaling].
        """
        return F.interpolate(
            x_L, scale_factor=self.SR_SCALE,
            mode='bicubic', align_corners=False
        )

    def _encode_image(self, x):
        """Encode image to latent space using VAE encoder."""
        if self.use_real_backbone:
            with torch.no_grad():
                latent_dist = self.vae.encode(self._normalize_vae_input(x)).latent_dist
                z = latent_dist.mean  # Deterministic encoding
                z = z * self.vae.config.scaling_factor
            return z
        else:
            return self.vae_encoder_fb(x)

    def _encode_text(self, batch_size, device):
        """Encode the fixed prompt using the text encoder."""
        if self.use_real_backbone:
            with torch.no_grad():
                if self.precomputed_text_context is not None:
                    context = self.precomputed_text_context.to(device).expand(batch_size, -1, -1)
                else:
                    inputs = self.tokenizer(
                        self.FIXED_PROMPT,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=256,
                        truncation=True,
                    ).to(device)
                    outputs = self.text_encoder(**inputs)
                    # ChatGLM variants may output [S, B, C] or [B, S, C].
                    context = self._normalize_single_prompt_context(
                        outputs.last_hidden_state, source="text encoder output"
                    )
                    # Expand fixed-prompt context to current batch size
                    context = context.expand(batch_size, -1, -1)
            return context
        else:
            return self._fallback_context.expand(batch_size, -1, -1).to(device)

    def _predict_noise(self, z_noisy, timesteps, context, text_token_indices):
        """UNet forward: predict noise and extract text attention maps."""
        if self.use_real_backbone:
            # Fix M1: Clear previous attention maps
            self.taca_manager.clear()

            # Forward through UNet — TACA processors capture attention maps
            noise_pred = self.unet_backbone(
                z_noisy, timesteps,
                encoder_hidden_states=context,
            ).sample

            # Fix M1: Aggregate attention maps from all M cross-attention layers
            a_tex = self.taca_manager.aggregate_attention(z_noisy.shape)
            return noise_pred, a_tex
        else:
            return self.unet_fb(z_noisy, timesteps, context, text_token_indices)

    def _get_schedule_params(self, t: int, device):
        """Get α_t, β_t coefficients for the one-step denoising formula.

        Paper Eq.1: ẑ_H = (1/√ᾱ_t) · z_{L,t} − (√(1-ᾱ_t)/√ᾱ_t) · n̂
        """
        alpha_bar_t = self.alpha_cumprod[t]
        coeff1 = 1.0 / torch.sqrt(alpha_bar_t)
        coeff2 = torch.sqrt(1.0 - alpha_bar_t) / torch.sqrt(alpha_bar_t)
        return coeff1.to(device), coeff2.to(device)

    def _add_noise(self, z, t: int):
        """Add noise at timestep t: z_t = √ᾱ_t · z + √(1-ᾱ_t) · ε"""
        alpha_bar_t = self.alpha_cumprod[t]
        noise = torch.randn_like(z)
        z_noisy = torch.sqrt(alpha_bar_t) * z + torch.sqrt(1.0 - alpha_bar_t) * noise
        return z_noisy, noise

    def forward_train(self, x_L, context=None, text_token_indices=None):
        """
        Training forward pass.

        Args:
            x_L:    LR image [B, 3, H, W]
            context: optional pre-computed text context [B, SeqLen, ContextDim]
                     Ignored when use_real_backbone=True (model encodes
                     the fixed prompt internally via ChatGLM).
            text_token_indices: list[int], positions of "text" token

        Returns:
            x_hr_pred:   predicted HR image [B, 3, H', W']
            s_mask_pred: predicted segmentation mask [B, 1, H', W']
        """
        B = x_L.shape[0]
        device = x_L.device

        # Use stored text_token_indices if not provided
        if text_token_indices is None:
            text_token_indices = self._text_token_indices

        # 1. Encode text (fixed prompt)
        # Fix T5: When using the real Kolors backbone, ALWAYS use the
        # model's own text encoder (ChatGLM) to encode the fixed prompt.
        # The Dataset provides a placeholder random context which is
        # incorrect for real training — ignore it.
        if self.use_real_backbone:
            context = self._encode_text(B, device)
        elif context is None:
            context = self._encode_text(B, device)

        # Fix M4: Upsample LR to HR resolution before VAE encoding
        x_L_up = self._upsample_lr(x_L)

        # 2. VAE encode (on upsampled LR image)
        z_L = self._encode_image(x_L_up)

        # 3. Add noise at fixed t=200
        t = self.FIXED_TIMESTEP
        z_L_noisy, noise = self._add_noise(z_L, t)

        # 4. UNet predicts noise + extracts text attention
        timesteps = torch.full((B,), t, dtype=torch.long, device=device)
        noise_pred, a_tex = self._predict_noise(
            z_L_noisy, timesteps, context, text_token_indices
        )

        # 5. One-step denoising (Paper Eq.1):
        #    ẑ_H = (1/√ᾱ_t) · z_{L,t} − (√(1-ᾱ_t)/√ᾱ_t) · n̂
        coeff1, coeff2 = self._get_schedule_params(t, device)
        z_H = coeff1 * z_L_noisy - coeff2 * noise_pred

        # 6. JSD: decode to image + segmentation mask
        x_hr_pred, s_mask_pred = self.jsd(z_H, a_tex)
        if self.use_real_backbone:
            x_hr_pred = self._denormalize_vae_output(x_hr_pred)

        return x_hr_pred, s_mask_pred

    def get_trainable_params(self):
        """Return only the parameters that should be trained.

        Paper: LoRA on UNet cross-attn + JSD (both decoders + CDIB)
               + TACA W_a projection
        Everything else (VAE encoder, text encoder) is frozen.
        """
        params = list(self.jsd.parameters())
        if self.use_real_backbone:
            # Only LoRA params from UNet
            params += [p for p in self.unet_backbone.parameters() if p.requires_grad]
            # TACA W_a projection
            if self.taca_manager is not None:
                params += list(self.taca_manager.parameters())
        else:
            params += list(self.unet_fb.parameters())
            params += list(self.vae_encoder_fb.parameters())
        return params
