#!/usr/bin/env python3
"""Single-process TADiSR inference entrypoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from models.tadisr_model import TADiSRWrapper


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
CHECKPOINT_VERSION = 2
LR_PAD_MULTIPLE = 16


def load_yaml_config(config_path: str | os.PathLike) -> dict:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format in {path}: expected YAML mapping")
    return cfg


def resolve_precision_dtype(precision: str) -> torch.dtype:
    precision = str(precision or "fp32").lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16", "half"}:
        return torch.float16
    if precision in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}. Use bf16, fp16, or fp32.")


def _is_torch_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    if not hasattr(torch, "npu"):
        return False
    is_available = getattr(torch.npu, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def resolve_device(device_name: str) -> torch.device:
    req = str(device_name or "cpu").lower()
    if req == "npu":
        if not _is_torch_npu_available():
            raise RuntimeError("Requested device=npu, but torch_npu is unavailable.")
        if hasattr(torch.npu, "set_device"):
            torch.npu.set_device(0)
        return torch.device("npu", 0)
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda, but CUDA is unavailable.")
        torch.cuda.set_device(0)
        return torch.device("cuda", 0)
    if req == "cpu":
        return torch.device("cpu")
    return torch.device(device_name)


def load_inference_checkpoint(model: TADiSRWrapper, checkpoint_path: str | os.PathLike, *, is_main: bool = True) -> dict:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Inference checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: expected dict payload in {path}")

    version = checkpoint.get("checkpoint_version", "legacy")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Invalid checkpoint: missing model_state_dict in {path}")

    load_result = model.load_state_dict(model_state, strict=False)
    if is_main:
        if version != CHECKPOINT_VERSION:
            print(f"[Infer] Warning: loading checkpoint version {version}; expected v{CHECKPOINT_VERSION}.")
        print(f"[Infer] Loaded checkpoint: {path}")
        print(f"[Infer] Model tensors loaded: {len(model_state)}")
        print(f"[Infer] Missing model keys: {len(load_result.missing_keys)}")
        print(f"[Infer] Unexpected model keys: {len(load_result.unexpected_keys)}")

    return {
        "checkpoint_version": version,
        "loaded_tensors": len(model_state),
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
    }


def build_model_from_config(
    cfg: dict,
    *,
    checkpoint_path: str | os.PathLike,
    pretrained_path: str | None = None,
    precomputed_text_context_path: str | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> TADiSRWrapper:
    use_kolors = bool(cfg.get("use_kolors", True))
    context_path = precomputed_text_context_path or cfg.get("precomputed_text_context_path")
    if use_kolors:
        if not context_path:
            raise ValueError(
                "Real Kolors inference requires precomputed_text_context_path. "
                "Pass --precomputed-text-context or set it in the config."
            )
        if not Path(context_path).is_file():
            raise FileNotFoundError(f"Precomputed text context not found: {context_path}")

    model_kwargs = {
        "use_kolors": use_kolors,
        "context_dim": int(cfg.get("context_dim", 4096)),
        "lora_rank": int(cfg.get("lora_rank", 16)),
        "jsd_dim": cfg.get("jsd_dim"),
        "gradient_checkpointing": False,
        "taca_query_chunk_size": int(cfg.get("taca_query_chunk_size", 1024) or 0),
        "taca_checkpoint": bool(cfg.get("taca_checkpoint", True)),
        "taca_detach": bool(cfg.get("taca_detach", False)),
        "fail_on_lora_error": bool(cfg.get("fail_on_lora_error", True)),
        "require_precomputed_text_context": use_kolors,
        "precomputed_text_context_path": context_path,
        "torch_dtype": dtype,
    }
    if pretrained_path:
        model_kwargs["pretrained_path"] = pretrained_path

    model = TADiSRWrapper(**model_kwargs)
    if use_kolors and not model.use_real_backbone:
        raise RuntimeError("Failed to initialize real Kolors backbone for inference.")

    load_inference_checkpoint(model, checkpoint_path)
    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    model.eval()
    return model


def collect_image_paths(input_path: str | os.PathLike) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if path.is_dir():
        images = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No supported images found in directory: {path}")
        return images
    raise FileNotFoundError(f"Input path not found: {path}")


def image_to_tensor(image_path: str | os.PathLike) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW/BCHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        array = tensor[0].mul(255.0).round().byte().numpy()
        return Image.fromarray(array)
    if tensor.shape[0] != 3:
        tensor = tensor[:3]
    array = tensor.permute(1, 2, 0).mul(255.0).round().byte().numpy()
    return Image.fromarray(array)


def pad_lr_tensor_to_even(tensor: torch.Tensor, *, multiple: int = 2) -> tuple[torch.Tensor, tuple[int, int]]:
    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(tensor.shape)}")
    h, w = tensor.shape[-2:]
    multiple = max(2, int(multiple))
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor, (h, w)


def crop_sr_to_original(tensor: torch.Tensor, original_lr_size: tuple[int, int], *, scale: int = 4) -> torch.Tensor:
    h, w = original_lr_size
    return tensor[..., : h * scale, : w * scale]


def taca_heatmap_to_image(a_tex: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    heatmap = a_tex.detach().float().abs().mean(dim=1, keepdim=True)
    flat = heatmap.flatten(start_dim=1)
    min_v = flat.amin(dim=1).view(-1, 1, 1, 1)
    max_v = flat.amax(dim=1).view(-1, 1, 1, 1)
    heatmap = (heatmap - min_v) / (max_v - min_v).clamp_min(1e-6)
    heatmap = F.interpolate(heatmap, size=output_size, mode="bilinear", align_corners=False)
    red = (1.5 * heatmap).clamp(0.0, 1.0)
    green = (1.5 - (2.0 * heatmap - 1.0).abs() * 1.5).clamp(0.0, 1.0)
    blue = (1.5 * (1.0 - heatmap)).clamp(0.0, 1.0)
    return torch.cat([red, green, blue], dim=1)


def resolve_output_paths(
    input_image: Path,
    output_path: str | os.PathLike,
    *,
    directory_mode: bool,
) -> tuple[Path, Path, Path]:
    output = Path(output_path)
    output_is_dir = directory_mode or output.suffix == "" or output.is_dir()
    if output_is_dir:
        output.mkdir(parents=True, exist_ok=True)
        sr_path = output / f"{input_image.stem}_sr.png"
        mask_path = output / f"{input_image.stem}_mask.png"
        debug_path = output / f"{input_image.stem}_taca.png"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        sr_path = output
        mask_path = output.with_name(f"{output.stem}_mask.png")
        debug_path = output.with_name(f"{output.stem}_taca.png")
    return sr_path, mask_path, debug_path


def run_image_inference(
    model: TADiSRWrapper,
    image_path: str | os.PathLike,
    *,
    device: torch.device,
    save_debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    lr = image_to_tensor(image_path)
    lr, original_size = pad_lr_tensor_to_even(lr, multiple=LR_PAD_MULTIPLE)
    lr = lr.to(device)

    with torch.inference_mode():
        output = model(lr, return_debug=save_debug)

    if save_debug:
        sr, mask, debug = output
    else:
        sr, mask = output
        debug = None

    scale = int(getattr(model, "SR_SCALE", 4))
    sr = crop_sr_to_original(sr, original_size, scale=scale)
    mask = crop_sr_to_original(mask, original_size, scale=scale)
    if debug is not None and debug.get("a_tex") is not None:
        debug = dict(debug)
        debug["output_size"] = tuple(sr.shape[-2:])
    return sr, mask, debug


def save_inference_outputs(
    sr: torch.Tensor,
    mask: torch.Tensor,
    debug: dict | None,
    *,
    sr_path: Path,
    mask_path: Path,
    debug_path: Path,
    save_mask: bool,
    save_debug: bool,
) -> None:
    tensor_to_pil(sr).save(sr_path)
    if save_mask:
        tensor_to_pil(mask).save(mask_path)
    if save_debug and debug is not None and debug.get("a_tex") is not None:
        heatmap = taca_heatmap_to_image(debug["a_tex"], debug["output_size"])
        tensor_to_pil(heatmap).save(debug_path)


def run_inference(args: argparse.Namespace) -> None:
    cfg = load_yaml_config(args.config)
    device = resolve_device(args.device or cfg.get("device", "npu"))
    dtype = resolve_precision_dtype(args.precision or cfg.get("precision", "bf16"))
    model = build_model_from_config(
        cfg,
        checkpoint_path=args.checkpoint,
        pretrained_path=args.pretrained_path,
        precomputed_text_context_path=args.precomputed_text_context,
        device=device,
        dtype=dtype,
    )

    images = collect_image_paths(args.input)
    directory_mode = Path(args.input).is_dir()
    for image_path in images:
        sr_path, mask_path, debug_path = resolve_output_paths(
            image_path,
            args.output,
            directory_mode=directory_mode,
        )
        sr, mask, debug = run_image_inference(
            model,
            image_path,
            device=device,
            save_debug=args.save_debug,
        )
        save_inference_outputs(
            sr,
            mask,
            debug,
            sr_path=sr_path,
            mask_path=mask_path,
            debug_path=debug_path,
            save_mask=args.save_mask,
            save_debug=args.save_debug,
        )
        print(f"[Infer] {image_path} -> {sr_path}")
        if args.save_mask:
            print(f"[Infer] mask -> {mask_path}")
        if args.save_debug:
            print(f"[Infer] debug -> {debug_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TADiSR single-image or directory inference")
    parser.add_argument("--input", required=True, help="Input image path or directory")
    parser.add_argument("--output", required=True, help="Output image path or directory")
    parser.add_argument("--config", default="configs/train/kolors.yaml", help="Training YAML used to build the model")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint path")
    parser.add_argument("--pretrained-path", default=None, help="Kolors base model path; defaults to TADiSRWrapper default")
    parser.add_argument("--precomputed-text-context", default=None, help="Fixed prompt embedding path; overrides config")
    parser.add_argument("--device", default="npu", help="Runtime device: npu, cuda, or cpu")
    parser.add_argument("--precision", default="bf16", help="Runtime precision: bf16, fp16, or fp32")
    parser.add_argument("--save-mask", action="store_true", default=True, help="Save predicted text mask")
    parser.add_argument("--no-save-mask", action="store_false", dest="save_mask", help="Do not save predicted text mask")
    parser.add_argument("--save-debug", action="store_true", help="Save TACA attention heatmap")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run_inference(parse_args(argv))


if __name__ == "__main__":
    main()
