"""
TADiSR training script.

Paper reference (Section 4.1):
  - Kolors variant of LDM
  - Fixed diffusion timestep t = 200
  - AdamW optimizer, base lr = 5e-5
  - 4× H20 GPUs, batch_size = 1/GPU, 200k iterations
  - 4× super-resolution
  - Training data: FTSR (45k) + Real-CE (337 pairs)
  - Total loss: ℓ_tot = ℓ_img + ℓ_seg (NO separate noise prediction loss)
"""
import os
import yaml
import argparse
import math
import random
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from models.tadisr_model import TADiSRWrapper
from models.vae_jsd import CDIB, DecoderBranch, JointSegmentationDecoders, UNetMidBlock2D, UpDecoderBlock2D
from losses.composite_loss import CompositeTADiSRLoss
from data.dataset import TADiSRDataset, tadisr_collate_fn


CHECKPOINT_VERSION = 2
CHECKPOINT_CONFIG_KEYS = (
    "use_kolors",
    "context_dim",
    "jsd_dim",
    "lora_rank",
    "lora_include_self_attention",
    "device",
    "dist_strategy",
    "precision",
    "batch_size",
    "lr",
    "lr_scheduler",
    "grad_clip",
    "warmup_iters",
    "min_lr_ratio",
    "max_iters",
    "epochs",
    "save_every",
    "ckpt_dir",
)


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_image(self, *args, **kwargs):
        return None

    def add_images(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def _tensorboard_scalar_value(value):
    if torch.is_tensor(value):
        return value.detach().float().item()
    return float(value)


def _log_tensorboard_loss_scalars(
    tb_writer,
    global_step,
    loss,
    loss_img,
    loss_seg,
    loss_components=None,
):
    tb_writer.add_scalar("train/loss", _tensorboard_scalar_value(loss), global_step)
    tb_writer.add_scalar("train/loss_img", _tensorboard_scalar_value(loss_img), global_step)
    tb_writer.add_scalar("train/loss_seg", _tensorboard_scalar_value(loss_seg), global_step)
    for name, value in (loss_components or {}).items():
        tb_writer.add_scalar(
            f"train/loss_components/{name}",
            _tensorboard_scalar_value(value),
            global_step,
        )


torch.jit._state.disable()

def load_config(config_path):
    """Load training config from a YAML file and return namespace args."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format in {config_path}: expected YAML mapping")

    required_keys = {
        "ftsr_dir", "realce_dir",
        "use_kolors", "context_dim", "jsd_dim", "lora_rank",
        "device", "batch_size", "lr", "grad_clip", "warmup_iters", "max_iters", "epochs",
        "log_every", "save_every", "ckpt_dir",
    }
    missing = required_keys.difference(cfg.keys())
    if missing:
        missing_txt = ", ".join(sorted(missing))
        raise ValueError(f"Missing config keys: {missing_txt}")

    # Optional: offline fixed-prompt embedding path to skip loading ChatGLM.
    cfg.setdefault("precomputed_text_context_path", None)
    # Distributed strategy: ddp | fsdp (used when WORLD_SIZE > 1)
    cfg.setdefault("dist_strategy", "ddp")
    cfg.setdefault("precision", "bf16")
    if "gradient_checkpointing" not in cfg and "use_gradient_checkpointing" in cfg:
        cfg["gradient_checkpointing"] = cfg["use_gradient_checkpointing"]
    cfg.setdefault("gradient_checkpointing", True)
    cfg.setdefault("fsdp_auto_wrap", True)
    cfg.setdefault("fsdp_activation_checkpointing", True)
    cfg.setdefault("lpips_resize", 512)
    cfg.setdefault("taca_query_chunk_size", 1024)
    cfg.setdefault("taca_checkpoint", True)
    cfg.setdefault("taca_detach", False)
    cfg.setdefault("fail_on_lora_error", True)
    cfg.setdefault("lora_include_self_attention", False)
    cfg.setdefault("require_precomputed_text_context", False)
    cfg.setdefault("tensorboard_image_every", 100)
    cfg.setdefault("tensorboard_image_max_samples", 1)
    cfg.setdefault("tensorboard_image_size", 256)
    cfg.setdefault("resume_from", None)
    cfg.setdefault("lr_scheduler", "constant")
    cfg.setdefault("min_lr_ratio", 0.1)

    return SimpleNamespace(**cfg)

def _is_torch_npu_available():
    """Best-effort check for torch_npu runtime availability."""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    if not hasattr(torch, "npu"):
        return False
    is_available = getattr(torch.npu, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return False


def _resolve_device(args_device, local_rank=0):
    """Resolve runtime device from args + current environment."""
    req = (args_device or "cpu").lower()

    if req == "npu":
        if _is_torch_npu_available():
            if hasattr(torch.npu, "set_device"):
                torch.npu.set_device(local_rank)
            return torch.device("npu", local_rank)
        print("[Device] Requested NPU but torch_npu is unavailable, fallback to CPU.")
        return torch.device("cpu")

    if req == "cuda":
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        print("[Device] Requested CUDA but CUDA is unavailable, fallback to CPU.")
        return torch.device("cpu")

    if req == "cpu":
        return torch.device("cpu")

    # Keep backward compatibility for users who pass any torch-recognized device.
    try:
        return torch.device(args_device)
    except Exception:
        print(f"[Device] Unknown device '{args_device}', fallback to CPU.")
        return torch.device("cpu")


def _resolve_dist_backend(device):
    if device.type == "cuda":
        return "nccl"
    if device.type == "npu":
        return "hccl"
    return "gloo"


def _resolve_fsdp_device_id(ddp_state):
    """Return FSDP device_id for current process, or None if unsupported."""
    device = ddp_state["device"]
    local_rank = ddp_state["local_rank"]

    if device.type == "cuda":
        return torch.cuda.current_device()

    if device.type == "npu":
        # torch_npu exposes current_device() in most environments; fallback to local_rank.
        current_device = getattr(torch.npu, "current_device", None)
        if callable(current_device):
            return current_device()
        return local_rank

    # CPU/gloo FSDP is not targeted in this project training path.
    return None


def _resolve_precision_dtype(precision: str):
    precision = str(precision or "fp32").lower()
    if precision in {"bf16", "bfloat16"}:
        return "bf16", torch.bfloat16
    if precision in {"fp16", "float16", "half"}:
        return "fp16", torch.float16
    if precision in {"fp32", "float32", "full"}:
        return "fp32", torch.float32
    raise ValueError(f"Unsupported precision: {precision}. Use bf16, fp16, or fp32.")


def _autocast_context(device, precision_name, dtype):
    if precision_name == "fp32":
        return nullcontext()
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def _build_fsdp_mixed_precision(precision_name, dtype):
    if precision_name == "fp32":
        return None
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
        cast_forward_inputs=True,
    )


def _is_large_checkpoint_target(module):
    if isinstance(module, (CDIB, DecoderBranch, JointSegmentationDecoders, UNetMidBlock2D, UpDecoderBlock2D)):
        return True
    name = module.__class__.__name__
    return name in {
        "BasicTransformerBlock",
        "CrossAttnDownBlock2D",
        "CrossAttnUpBlock2D",
        "DownBlock2D",
        "UpBlock2D",
        "UNetMidBlock2DCrossAttn",
    }


def _fsdp_auto_wrap_policy(module, recurse, nonwrapped_numel):
    if recurse:
        return True
    return _is_large_checkpoint_target(module)


def _apply_fsdp_activation_checkpointing(model):
    def check_fn(module):
        return module.__class__.__name__ in {
            "BasicTransformerBlock",
            "CrossAttnDownBlock2D",
            "CrossAttnUpBlock2D",
            "DownBlock2D",
            "UpBlock2D",
            "UNetMidBlock2DCrossAttn",
        }

    matched = sum(1 for module in model.modules() if check_fn(module))
    if matched == 0:
        return 0

    def wrapper(module):
        return checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=check_fn,
    )
    return matched


def _print_trainable_summary(model, is_main):
    if not is_main:
        return
    raw_model = model.module if hasattr(model, "module") else model
    summary_fn = getattr(raw_model, "get_trainable_param_summary", None)
    if callable(summary_fn):
        summary = summary_fn()
        print("  Trainable parameter groups:")
        for name, count in summary.items():
            if count:
                print(f"    {name}: {count:,}")


def _cuda_memory_stats(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_mb": torch.cuda.memory_allocated(idx) / 1024 ** 2,
        "reserved_mb": torch.cuda.memory_reserved(idx) / 1024 ** 2,
        "peak_mb": torch.cuda.max_memory_allocated(idx) / 1024 ** 2,
    }


def _should_log_tensorboard_images(args, step: int) -> bool:
    every = int(getattr(args, "tensorboard_image_every", 100) or 0)
    if every <= 0:
        return False
    return step == 1 or step % every == 0


def _tensorboard_image_size(args) -> int:
    return max(1, int(getattr(args, "tensorboard_image_size", 256) or 256))


def _tensorboard_image_max_samples(args, batch_size: int) -> int:
    configured = int(getattr(args, "tensorboard_image_max_samples", 1) or 1)
    return max(1, min(configured, batch_size))


def _as_image_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW image tensor, got shape {tuple(tensor.shape)}")
    return tensor.detach().float()


def _resize_image_batch(batch: torch.Tensor, size: int) -> torch.Tensor:
    if batch.shape[-2:] == (size, size):
        return batch
    return F.interpolate(batch, size=(size, size), mode="bilinear", align_corners=False)


def _normalize_per_sample(batch: torch.Tensor) -> torch.Tensor:
    flat = batch.flatten(start_dim=1)
    min_v = flat.amin(dim=1).view(-1, 1, 1, 1)
    max_v = flat.amax(dim=1).view(-1, 1, 1, 1)
    return (batch - min_v) / (max_v - min_v).clamp_min(1e-6)


def _to_rgb(batch: torch.Tensor) -> torch.Tensor:
    if batch.shape[1] == 3:
        return batch
    if batch.shape[1] == 1:
        return batch.repeat(1, 3, 1, 1)
    if batch.shape[1] > 3:
        return batch[:, :3]
    raise ValueError(f"Cannot convert {batch.shape[1]} channels to RGB")


def _heatmap_to_rgb(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap.clamp(0.0, 1.0)
    red = (1.5 * heatmap).clamp(0.0, 1.0)
    green = (1.5 - (2.0 * heatmap - 1.0).abs() * 1.5).clamp(0.0, 1.0)
    blue = (1.5 * (1.0 - heatmap)).clamp(0.0, 1.0)
    return torch.cat([red, green, blue], dim=1)


def _prepare_display_batch(
    tensor: torch.Tensor,
    size: int,
    max_samples: int,
    *,
    normalize: bool = False,
    rgb: bool = True,
) -> torch.Tensor:
    batch = _as_image_batch(tensor)[:max_samples]
    batch = _normalize_per_sample(batch) if normalize else batch.clamp(0.0, 1.0)
    batch = _resize_image_batch(batch, size)
    if rgb:
        batch = _to_rgb(batch)
    return batch.cpu()


def _prepare_taca_heatmap(
    a_tex: torch.Tensor,
    size: int,
    max_samples: int,
) -> torch.Tensor:
    heatmap = _as_image_batch(a_tex)[:max_samples].abs().mean(dim=1, keepdim=True)
    heatmap = _normalize_per_sample(heatmap)
    heatmap = _resize_image_batch(heatmap, size)
    return _heatmap_to_rgb(heatmap).cpu()


def _make_image_grid(batch: torch.Tensor, nrow=None, padding: int = 2) -> torch.Tensor:
    if batch.ndim == 3:
        return batch
    if batch.ndim != 4:
        raise ValueError(f"Expected BCHW tensor for image grid, got shape {tuple(batch.shape)}")

    batch_size, channels, height, width = batch.shape
    if batch_size == 1:
        return batch[0]

    nrow = min(batch_size, int(nrow or batch_size))
    ncol = (batch_size + nrow - 1) // nrow
    grid_h = ncol * height + (ncol - 1) * padding
    grid_w = nrow * width + (nrow - 1) * padding
    grid = batch.new_full((channels, grid_h, grid_w), 1.0)

    for idx in range(batch_size):
        row = idx // nrow
        col = idx % nrow
        top = row * (height + padding)
        left = col * (width + padding)
        grid[:, top:top + height, left:left + width] = batch[idx]
    return grid


def _make_overview_grid(panels, padding: int = 2) -> torch.Tensor:
    if not panels:
        raise ValueError("At least one image panel is required")

    batch_size = panels[0].shape[0]
    _, channels, height, _ = panels[0].shape
    rows = []
    for sample_idx in range(batch_size):
        row_parts = []
        for panel_idx, panel in enumerate(panels):
            if panel_idx > 0:
                row_parts.append(panel.new_full((channels, height, padding), 1.0))
            row_parts.append(panel[sample_idx])
        rows.append(torch.cat(row_parts, dim=2))

    if len(rows) == 1:
        return rows[0]

    row_width = rows[0].shape[-1]
    stacked = []
    for row_idx, row in enumerate(rows):
        if row_idx > 0:
            stacked.append(row.new_full((channels, padding, row_width), 1.0))
        stacked.append(row)
    return torch.cat(stacked, dim=1)


def _add_tensorboard_image(tb_writer, tag: str, batch: torch.Tensor, step: int):
    tb_writer.add_image(tag, _make_image_grid(batch), step, dataformats="CHW")


def _log_tensorboard_images(
    tb_writer,
    args,
    step: int,
    lr_img: torch.Tensor,
    hr_img: torch.Tensor,
    x_pred: torch.Tensor,
    s_pred: torch.Tensor,
    mask: torch.Tensor,
    debug,
):
    size = _tensorboard_image_size(args)
    max_samples = _tensorboard_image_max_samples(args, lr_img.shape[0])

    lr_vis = _prepare_display_batch(lr_img, size, max_samples)
    hr_vis = _prepare_display_batch(hr_img, size, max_samples)
    pred_vis = _prepare_display_batch(x_pred, size, max_samples)
    pred_mask_vis = _prepare_display_batch(s_pred, size, max_samples)
    mask_vis = _prepare_display_batch(mask, size, max_samples)

    _add_tensorboard_image(tb_writer, "images/lr_upsampled", lr_vis, step)
    _add_tensorboard_image(tb_writer, "images/hr_gt", hr_vis, step)
    _add_tensorboard_image(tb_writer, "images/jsd_pred_hr", pred_vis, step)
    _add_tensorboard_image(tb_writer, "images/jsd_pred_mask", pred_mask_vis, step)
    _add_tensorboard_image(tb_writer, "images/mask_gt", mask_vis, step)

    overview_panels = [lr_vis, hr_vis, pred_vis, pred_mask_vis, mask_vis]
    if debug is not None and debug.get("a_tex") is not None:
        taca_vis = _prepare_taca_heatmap(debug["a_tex"], size, max_samples)
        _add_tensorboard_image(tb_writer, "images/taca_attention", taca_vis, step)
        overview_panels.append(taca_vis)

    tb_writer.add_image(
        "images/overview_grid",
        _make_overview_grid(overview_panels),
        step,
        dataformats="CHW",
    )


def _checkpoint_config_summary(args):
    return {key: getattr(args, key, None) for key in CHECKPOINT_CONFIG_KEYS}


def _checkpoint_trainable_state_dict(raw_model, trainable_params):
    """Return the compact checkpoint payload for LoRA/JSD/TACA training."""
    trainable_ids = {id(p) for p in trainable_params}
    save_keys = set()
    for name, param in raw_model.named_parameters():
        if id(param) in trainable_ids:
            save_keys.add(name)
        elif "jsd" in name or "taca" in name:
            save_keys.add(name)

    model_sd = raw_model.state_dict()
    return {k: v for k, v in model_sd.items() if k in save_keys}


def _capture_rng_state(device):
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "npu" and hasattr(torch, "npu"):
        get_rng_state_all = getattr(torch.npu, "get_rng_state_all", None)
        if callable(get_rng_state_all):
            rng_state["npu"] = get_rng_state_all()
    return rng_state


def _restore_rng_state(rng_state, device, *, is_main=True):
    if not rng_state:
        if is_main:
            print("  Resume warning: checkpoint has no RNG state; continuing best-effort.")
        return

    try:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        if "npu" in rng_state and device.type == "npu" and hasattr(torch, "npu"):
            set_rng_state_all = getattr(torch.npu, "set_rng_state_all", None)
            if callable(set_rng_state_all):
                set_rng_state_all(rng_state["npu"])
    except Exception as exc:
        if is_main:
            print(f"  Resume warning: failed to restore RNG state ({exc}); continuing best-effort.")


def _build_training_checkpoint(
    raw_model,
    trainable_params,
    optimizer,
    scheduler,
    scaler,
    args,
    *,
    global_step,
    epoch,
    next_batch_idx,
    loss_value,
    device,
):
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": int(global_step),
        "epoch": int(epoch),
        "next_batch_idx": int(next_batch_idx),
        "model_state_dict": _checkpoint_trainable_state_dict(raw_model, trainable_params),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": _capture_rng_state(device),
        "training_config": _checkpoint_config_summary(args),
        "loss": float(loss_value),
    }


def _infer_resume_position(checkpoint, steps_per_epoch, *, is_main=True):
    global_step = int(checkpoint.get("step", 0) or 0)
    has_position = "epoch" in checkpoint and "next_batch_idx" in checkpoint
    if has_position:
        return global_step, int(checkpoint["epoch"]), int(checkpoint["next_batch_idx"])

    if steps_per_epoch and steps_per_epoch > 0:
        start_epoch = global_step // steps_per_epoch
        start_batch_idx = global_step % steps_per_epoch
    else:
        start_epoch = 0
        start_batch_idx = 0

    if is_main:
        print(
            "  Resume warning: checkpoint has no epoch/next_batch_idx; "
            f"inferred epoch={start_epoch}, batch={start_batch_idx} from step={global_step}."
        )
    return global_step, start_epoch, start_batch_idx


def load_training_checkpoint(
    ckpt_path,
    raw_model,
    optimizer,
    scheduler,
    scaler,
    device,
    *,
    steps_per_epoch=None,
    is_main=True,
):
    if not ckpt_path:
        return {"global_step": 0, "start_epoch": 0, "start_batch_idx": 0}
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Invalid checkpoint: missing model_state_dict in {ckpt_path}")

    load_result = raw_model.load_state_dict(model_state, strict=False)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif is_main:
        print("  Resume warning: checkpoint has no optimizer_state_dict.")

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    elif scheduler is not None and is_main:
        print("  Resume warning: checkpoint has no scheduler_state_dict.")

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    elif scaler is not None and scaler.is_enabled() and is_main:
        print("  Resume warning: checkpoint has no scaler_state_dict.")

    _restore_rng_state(checkpoint.get("rng_state"), device, is_main=is_main)
    global_step, start_epoch, start_batch_idx = _infer_resume_position(
        checkpoint,
        steps_per_epoch,
        is_main=is_main,
    )

    if is_main:
        version = checkpoint.get("checkpoint_version", "legacy")
        print(
            f"  Resumed checkpoint v{version} from {ckpt_path}: "
            f"step={global_step}, epoch={start_epoch}, next_batch_idx={start_batch_idx}"
        )
        if load_result.missing_keys:
            print(
                f"  Resume note: {len(load_result.missing_keys)} missing model keys "
                "(expected for compact trainable-only checkpoints)."
            )
        if load_result.unexpected_keys:
            print(f"  Resume note: {len(load_result.unexpected_keys)} unexpected model keys.")

    return {
        "global_step": global_step,
        "start_epoch": start_epoch,
        "start_batch_idx": start_batch_idx,
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
    }


def save_training_checkpoint(
    ckpt_path,
    raw_model,
    trainable_params,
    optimizer,
    scheduler,
    scaler,
    args,
    *,
    global_step,
    epoch,
    next_batch_idx,
    loss_value,
    device,
):
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    checkpoint = _build_training_checkpoint(
        raw_model,
        trainable_params,
        optimizer,
        scheduler,
        scaler,
        args,
        global_step=global_step,
        epoch=epoch,
        next_batch_idx=next_batch_idx,
        loss_value=loss_value,
        device=device,
    )
    torch.save(checkpoint, ckpt_path)
    return len(checkpoint["model_state_dict"])


def _normalize_lr_scheduler_name(name):
    scheduler_name = str(name or "constant").strip().lower().replace("_", "-")
    aliases = {
        "none": "none",
        "off": "none",
        "disabled": "none",
        "disable": "none",
        "constant": "constant",
        "fixed": "constant",
        "warmup": "constant",
        "cosine": "cosine",
        "warmup-cosine": "cosine",
    }
    if scheduler_name not in aliases:
        valid = ", ".join(sorted({"none", "constant", "cosine"}))
        raise ValueError(f"Unsupported lr_scheduler: {name!r}. Use one of: {valid}.")
    return aliases[scheduler_name]


def _lr_multiplier_for_step(step, *, scheduler_name, warmup_iters, max_iters, min_lr_ratio):
    step = max(0, int(step))
    warmup_iters = int(warmup_iters or 0)
    max_iters = int(max_iters or 0)
    min_lr_ratio = float(min_lr_ratio)

    if warmup_iters < 0:
        raise ValueError("warmup_iters must be >= 0.")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1].")

    if warmup_iters > 0 and step < warmup_iters:
        return float(step + 1) / float(warmup_iters)

    if scheduler_name == "constant":
        return 1.0

    if scheduler_name == "cosine":
        if max_iters <= 0:
            raise ValueError("max_iters must be > 0 for cosine lr scheduling.")
        decay_iters = max(1, max_iters - warmup_iters)
        decay_step = min(max(step - warmup_iters, 0), decay_iters)
        progress = float(decay_step) / float(decay_iters)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    raise ValueError(f"Unsupported normalized lr scheduler: {scheduler_name!r}.")


def build_lr_scheduler(optimizer, args):
    """Build the configured LR scheduler, or None for a strictly fixed LR."""
    scheduler_name = _normalize_lr_scheduler_name(getattr(args, "lr_scheduler", "constant"))
    warmup_iters = int(getattr(args, "warmup_iters", 0) or 0)
    max_iters = int(getattr(args, "max_iters", 0) or 0)
    min_lr_ratio = float(getattr(args, "min_lr_ratio", 0.1))

    if scheduler_name == "none":
        return None
    if scheduler_name == "constant" and warmup_iters <= 0:
        return None

    def lr_lambda(step):
        return _lr_multiplier_for_step(
            step,
            scheduler_name=scheduler_name,
            warmup_iters=warmup_iters,
            max_iters=max_iters,
            min_lr_ratio=min_lr_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_datasets(args):
    """Create training datasets: FTSR + optionally Real-CE."""
    datasets = []

    # FTSR dataset
    if os.path.isdir(args.ftsr_dir):
        ftsr = TADiSRDataset(
            data_root=args.ftsr_dir,
            context_dim=args.context_dim,
        )
        if len(ftsr) > 0:
            print(f"[Data] FTSR: {len(ftsr)} samples from {args.ftsr_dir}")
            datasets.append(ftsr)

    # Real-CE dataset (paper: 337 training pairs)
    if args.realce_dir and os.path.isdir(args.realce_dir):
        realce = TADiSRDataset(
            data_root=args.realce_dir,
            context_dim=args.context_dim,
        )
        if len(realce) > 0:
            print(f"[Data] Real-CE: {len(realce)} samples from {args.realce_dir}")
            datasets.append(realce)
    else:
        print(f"[Data] Real-CE not found at {args.realce_dir} (optional)")

    if not datasets:
        searched = [f"FTSR={args.ftsr_dir}"]
        if args.realce_dir:
            searched.append(f"Real-CE={args.realce_dir}")
        raise FileNotFoundError(
            "No training data found. Prepare FTSR and optionally Real-CE before "
            "starting training; searched " + ", ".join(searched)
        )

    if len(datasets) > 1:
        return ConcatDataset(datasets)
    return datasets[0]


def setup_distributed(args):
    """Initialize DDP from torchrun env vars if available."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if not distributed:
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": _resolve_device(args.device, local_rank=0),
        }

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _resolve_device(args.device, local_rank=local_rank)

    if not dist.is_initialized():
        backend = _resolve_dist_backend(device)
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()

    return {
        "distributed": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def cleanup_distributed(ddp_state):
    if ddp_state["distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    dist_strategy = str(getattr(args, "dist_strategy", "ddp")).lower()
    if dist_strategy not in {"ddp", "fsdp"}:
        raise ValueError(f"Unsupported dist_strategy: {dist_strategy}. Use 'ddp' or 'fsdp'.")
    resume_from = getattr(args, "resume_from", None)
    if resume_from and dist_strategy == "fsdp":
        raise NotImplementedError(
            "Checkpoint resume currently supports single-process/DDP only; "
            "FSDP resume needs a full-state-dict checkpoint path and is not implemented yet."
        )
    ddp_state = setup_distributed(args)
    is_main = ddp_state["rank"] == 0
    device = ddp_state["device"]
    precision_name, precision_dtype = _resolve_precision_dtype(getattr(args, "precision", "bf16"))

    if is_main:
        print(f"Starting TADiSR training on {device}...")
        print(f"  LR: {args.lr}, Batch size(per-process): {args.batch_size}")
        print(
            f"  LR scheduler: {getattr(args, 'lr_scheduler', 'constant')} "
            f"(warmup={getattr(args, 'warmup_iters', 0)}, "
            f"min_lr_ratio={getattr(args, 'min_lr_ratio', 0.1)})"
        )
        print(f"  Max iterations: {args.max_iters}")
        print(f"  Gradient clip: {args.grad_clip}")
        print(f"  Precision: {precision_name}")
        print(f"  Gradient checkpointing: {bool(getattr(args, 'gradient_checkpointing', False))}")
        print(f"  TACA chunk: {getattr(args, 'taca_query_chunk_size', 0)}")
        print(f"  TACA checkpoint: {bool(getattr(args, 'taca_checkpoint', False))}")
        print(f"  LPIPS resize: {getattr(args, 'lpips_resize', 0)}")
        print(f"  LoRA include attn1: {bool(getattr(args, 'lora_include_self_attention', False))}")
        if ddp_state["distributed"]:
            print(f"  Distributed enabled ({dist_strategy.upper()}) — world_size: {ddp_state['world_size']}")

    tb_log_dir = getattr(args, "tensorboard_log_dir", os.path.join(args.ckpt_dir, "tensorboard"))
    if is_main and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
    elif is_main:
        tb_writer = _NullSummaryWriter()
        print("  TensorBoard not installed; scalar logging disabled.")
    else:
        tb_writer = None
    if is_main:
        print(f"  TensorBoard log dir: {tb_log_dir}")

    # 1. Dataset
    dataset = create_datasets(args)
    sampler = None
    shuffle = True
    if ddp_state["distributed"]:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_state["world_size"],
            rank=ddp_state["rank"],
            shuffle=True,
            drop_last=True,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=tadisr_collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # 2. Model
    model = TADiSRWrapper(
        use_kolors=args.use_kolors,
        context_dim=args.context_dim,
        lora_rank=args.lora_rank,
        jsd_dim=getattr(args, "jsd_dim", None),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        taca_query_chunk_size=getattr(args, "taca_query_chunk_size", 1024),
        taca_checkpoint=getattr(args, "taca_checkpoint", True),
        taca_detach=getattr(args, "taca_detach", False),
        fail_on_lora_error=getattr(args, "fail_on_lora_error", True),
        lora_include_self_attention=getattr(args, "lora_include_self_attention", False),
        require_precomputed_text_context=getattr(args, "require_precomputed_text_context", False),
        precomputed_text_context_path=getattr(args, "precomputed_text_context_path", None),
        torch_dtype=precision_dtype,
    )
    if not (ddp_state["distributed"] and dist_strategy == "fsdp"):
        model = model.to(device)

    # 3. Optimizer — only trainable parameters (LoRA + JSD)
    model.assert_trainable_parameter_contract()
    trainable_params = model.get_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in trainable_params)
    if is_main:
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {train_params:,} ({100*train_params/total_params:.2f}%)")
    _print_trainable_summary(model, is_main)

    fsdp_checkpoint_targets = 0
    if (
        ddp_state["distributed"]
        and dist_strategy == "fsdp"
        and getattr(args, "fsdp_activation_checkpointing", False)
    ):
        fsdp_checkpoint_targets = _apply_fsdp_activation_checkpointing(model)


    model = model.to(precision_dtype)
    if ddp_state["distributed"]:
        if dist_strategy == "ddp":
            model = DDP(
                model,
                device_ids=[ddp_state["local_rank"]] if device.type in {"cuda", "npu"} else None,
                output_device=ddp_state["local_rank"] if device.type in {"cuda", "npu"} else None,
                find_unused_parameters=False,
            )
        else:
            fsdp_device_id = _resolve_fsdp_device_id(ddp_state)
            if fsdp_device_id is None:
                raise RuntimeError(
                    "FSDP requires accelerator device in this project "
                    f"(cuda/npu), got device={device.type}."
                )
            model = FSDP(
                model,
                device_id=fsdp_device_id,
                use_orig_params=True,
                auto_wrap_policy=_fsdp_auto_wrap_policy if getattr(args, "fsdp_auto_wrap", True) else None,
                mixed_precision=_build_fsdp_mixed_precision(precision_name, precision_dtype),
            )
    raw_model = model.module if ddp_state["distributed"] else model
    if is_main and ddp_state["distributed"] and dist_strategy == "fsdp":
        print(f"  FSDP auto wrap: {bool(getattr(args, 'fsdp_auto_wrap', True))}")
        print(f"  FSDP activation checkpoint targets: {fsdp_checkpoint_targets}")

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(precision_name == "fp16" and device.type == "cuda"),
    )

    # 4. Learning rate scheduler (optional warmup + cosine decay)
    scheduler = build_lr_scheduler(optimizer, args)

    # 5. Loss
    criterion = CompositeTADiSRLoss(
        lpips_resize=getattr(args, "lpips_resize", 0),
    ).to(device)

    resume_state = {"global_step": 0, "start_epoch": 0, "start_batch_idx": 0}
    if resume_from:
        resume_state = load_training_checkpoint(
            resume_from,
            raw_model,
            optimizer,
            scheduler,
            scaler,
            device,
            steps_per_epoch=len(loader),
            is_main=is_main,
        )
        if ddp_state["distributed"]:
            dist.barrier()

    # 6. Training Loop
    model.train()
    global_step = resume_state["global_step"]
    start_epoch = resume_state["start_epoch"]
    start_batch_idx = resume_state["start_batch_idx"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for ep in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)

        for step, batch in enumerate(loader):
            if ep == start_epoch and step < start_batch_idx:
                continue
            if global_step >= args.max_iters:
                break

            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            mask = batch["mask"].to(device)
            context = batch["context"].to(device)
            text_indices = batch["text_indices"][0]
            log_tb_images = is_main and _should_log_tensorboard_images(args, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, precision_name, precision_dtype):
                # Forward — Paper loss: ℓ_tot = ℓ_img + ℓ_seg (NO noise loss)
                model_out = model.forward(
                    lr_img, context, text_indices, return_debug=log_tb_images
                )
                if log_tb_images:
                    x_pred, s_pred, debug = model_out
                else:
                    x_pred, s_pred = model_out
                    debug = None

                # Calculate composite loss
                loss, loss_img, loss_seg, loss_components = criterion(
                    x_pred.float(),
                    hr_img.float(),
                    s_pred.float(),
                    mask.float(),
                    return_components=True,
                )
            loss_for_backward = loss.float()

            # Backward with gradient clipping
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    if dist_strategy == "fsdp" and isinstance(model, FSDP):
                        model.clip_grad_norm_(args.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            args.grad_clip,
                        )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_for_backward.backward()
                if args.grad_clip > 0:
                    if dist_strategy == "fsdp" and isinstance(model, FSDP):
                        model.clip_grad_norm_(args.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            args.grad_clip,
                        )
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            global_step += 1

            if log_tb_images:
                _log_tensorboard_images(
                    tb_writer,
                    args,
                    global_step,
                    lr_img,
                    hr_img,
                    x_pred,
                    s_pred,
                    mask,
                    debug,
                )

            if is_main and global_step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                _log_tensorboard_loss_scalars(
                    tb_writer,
                    global_step,
                    loss,
                    loss_img,
                    loss_seg,
                    loss_components,
                )
                tb_writer.add_scalar("train/lr", current_lr, global_step)
                mem = _cuda_memory_stats(device)
                if mem is not None:
                    tb_writer.add_scalar("memory/allocated_mb", mem["allocated_mb"], global_step)
                    tb_writer.add_scalar("memory/reserved_mb", mem["reserved_mb"], global_step)
                    tb_writer.add_scalar("memory/peak_mb", mem["peak_mb"], global_step)
                    mem_txt = (
                        f" Mem: alloc={mem['allocated_mb']:.0f}MB "
                        f"reserved={mem['reserved_mb']:.0f}MB peak={mem['peak_mb']:.0f}MB"
                    )
                else:
                    mem_txt = ""
                print(
                    f"[Step {global_step:6d}] "
                    f"Loss: {loss.item():.4f} "
                    f"(Img: {loss_img.item():.4f}, "
                    f"Seg: {loss_seg.item():.4f}) "
                    f"LR: {current_lr:.2e}"
                    f"{mem_txt}"
                )

            # Save checkpoint
            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"tadisr_step{global_step}.pt")
                saved_tensors = save_training_checkpoint(
                    ckpt_path,
                    raw_model,
                    trainable_params,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                    global_step=global_step,
                    epoch=ep,
                    next_batch_idx=step + 1,
                    loss_value=loss.item(),
                    device=device,
                )
                print(f"  Checkpoint saved to {ckpt_path} ({saved_tensors} tensors)")

        if global_step >= args.max_iters:
            break

    if is_main:
        mem = _cuda_memory_stats(device)
        if mem is not None:
            print(
                f"Peak CUDA memory: alloc={mem['allocated_mb']:.0f}MB "
                f"reserved={mem['reserved_mb']:.0f}MB peak={mem['peak_mb']:.0f}MB"
            )
        tb_writer.close()
        print(f"\nTraining complete! {global_step} iterations.")
    cleanup_distributed(ddp_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TADiSR Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/default.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional checkpoint path that overrides resume_from in the config.",
    )

    cli_args = parser.parse_args()
    args = load_config(cli_args.config)
    if cli_args.resume_from is not None:
        args.resume_from = cli_args.resume_from
    train(args)
