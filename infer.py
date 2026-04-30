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
    if not use_kolors:
        raise ValueError(
            "Inference requires use_kolors=true. "
            "The non-Kolors runtime has been removed from production code."
        )
    context_path = precomputed_text_context_path or cfg.get("precomputed_text_context_path")
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


def validate_tile_args(tile_size: int, tile_overlap: int) -> tuple[int, int]:
    tile_size = int(tile_size or 0)
    tile_overlap = int(tile_overlap or 0)
    if tile_size < 0:
        raise ValueError("--tile-size must be >= 0")
    if tile_overlap < 0:
        raise ValueError("--tile-overlap must be >= 0")
    if tile_size == 0:
        return tile_size, tile_overlap
    if tile_size < LR_PAD_MULTIPLE:
        raise ValueError(f"--tile-size must be at least {LR_PAD_MULTIPLE} LR pixels")
    if tile_size % LR_PAD_MULTIPLE != 0:
        raise ValueError(f"--tile-size must be a multiple of {LR_PAD_MULTIPLE}")
    if tile_overlap >= tile_size:
        raise ValueError("--tile-overlap must be smaller than --tile-size")
    return tile_size, tile_overlap


def tile_starts(length: int, tile_size: int, tile_overlap: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"Invalid image dimension: {length}")
    if tile_size <= 0 or length <= tile_size:
        return [0]

    stride = tile_size - tile_overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_tile_regions(
    height: int,
    width: int,
    *,
    tile_size: int,
    tile_overlap: int,
) -> list[tuple[int, int, int, int]]:
    tile_size, tile_overlap = validate_tile_args(tile_size, tile_overlap)
    if tile_size <= 0:
        return [(0, height, 0, width)]

    y_starts = tile_starts(height, tile_size, tile_overlap)
    x_starts = tile_starts(width, tile_size, tile_overlap)
    return [
        (y0, min(y0 + tile_size, height), x0, min(x0 + tile_size, width))
        for y0 in y_starts
        for x0 in x_starts
    ]


def _edge_ramp(length: int, *, ascending: bool, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, device=device)
    ramp = torch.linspace(0.0, 1.0, steps=length + 2, device=device)[1:-1]
    return ramp if ascending else ramp.flip(0)


def build_tile_blend_weight(
    *,
    output_height: int,
    output_width: int,
    region: tuple[int, int, int, int],
    full_lr_size: tuple[int, int],
    tile_overlap: int,
    scale: int,
    device: torch.device,
) -> torch.Tensor:
    y0, y1, x0, x1 = region
    full_h, full_w = full_lr_size
    weight_y = torch.ones(output_height, device=device)
    weight_x = torch.ones(output_width, device=device)
    overlap = max(0, int(tile_overlap)) * int(scale)

    if overlap > 0:
        top = min(overlap, output_height)
        left = min(overlap, output_width)
        bottom = min(overlap, output_height)
        right = min(overlap, output_width)
        if y0 > 0:
            weight_y[:top] = _edge_ramp(top, ascending=True, device=device)
        if y1 < full_h:
            weight_y[-bottom:] = torch.minimum(
                weight_y[-bottom:],
                _edge_ramp(bottom, ascending=False, device=device),
            )
        if x0 > 0:
            weight_x[:left] = _edge_ramp(left, ascending=True, device=device)
        if x1 < full_w:
            weight_x[-right:] = torch.minimum(
                weight_x[-right:],
                _edge_ramp(right, ascending=False, device=device),
            )

    return weight_y.view(1, 1, output_height, 1) * weight_x.view(1, 1, 1, output_width)


def _forward_model(
    model: TADiSRWrapper,
    lr: torch.Tensor,
    *,
    save_debug: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    with torch.inference_mode():
        output = model(lr, return_debug=save_debug)

    if save_debug:
        sr, mask, debug = output
    else:
        sr, mask = output
        debug = None
    return sr, mask, debug


def run_tiled_model_inference(
    model: TADiSRWrapper,
    lr: torch.Tensor,
    *,
    device: torch.device,
    save_debug: bool,
    tile_size: int,
    tile_overlap: int,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    tile_size, tile_overlap = validate_tile_args(tile_size, tile_overlap)
    if lr.ndim != 4 or lr.shape[0] != 1:
        raise ValueError(f"Tiled inference expects one BCHW image, got shape {tuple(lr.shape)}")

    scale = int(getattr(model, "SR_SCALE", 4))
    _, channels, lr_h, lr_w = lr.shape
    regions = iter_tile_regions(lr_h, lr_w, tile_size=tile_size, tile_overlap=tile_overlap)
    out_h, out_w = lr_h * scale, lr_w * scale
    acc_device = torch.device("cpu")
    sr_accum = torch.zeros((1, channels, out_h, out_w), dtype=torch.float32, device=acc_device)
    mask_accum = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=acc_device)
    debug_accum = torch.zeros((1, 3, out_h, out_w), dtype=torch.float32, device=acc_device) if save_debug else None
    weight_accum = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=acc_device)

    for idx, region in enumerate(regions, start=1):
        y0, y1, x0, x1 = region
        tile = lr[..., y0:y1, x0:x1].to(device)
        sr_tile, mask_tile, debug_tile = _forward_model(model, tile, save_debug=save_debug)
        tile_lr_size = (y1 - y0, x1 - x0)
        sr_tile = crop_sr_to_original(sr_tile, tile_lr_size, scale=scale).detach().float().cpu()
        mask_tile = crop_sr_to_original(mask_tile, tile_lr_size, scale=scale).detach().float().cpu()

        hy0, hy1 = y0 * scale, y1 * scale
        hx0, hx1 = x0 * scale, x1 * scale
        weight = build_tile_blend_weight(
            output_height=hy1 - hy0,
            output_width=hx1 - hx0,
            region=region,
            full_lr_size=(lr_h, lr_w),
            tile_overlap=tile_overlap,
            scale=scale,
            device=acc_device,
        )
        sr_accum[..., hy0:hy1, hx0:hx1] += sr_tile * weight
        mask_accum[..., hy0:hy1, hx0:hx1] += mask_tile * weight
        weight_accum[..., hy0:hy1, hx0:hx1] += weight

        if debug_accum is not None and debug_tile is not None and debug_tile.get("a_tex") is not None:
            heatmap = taca_heatmap_to_image(debug_tile["a_tex"], (hy1 - hy0, hx1 - hx0)).detach().float().cpu()
            debug_accum[..., hy0:hy1, hx0:hx1] += heatmap * weight

        print(f"[Infer] tile {idx}/{len(regions)}: y={y0}:{y1}, x={x0}:{x1}")

    denom = weight_accum.clamp_min(1e-6)
    sr = sr_accum / denom
    mask = mask_accum / denom
    debug = None
    if debug_accum is not None:
        debug = {
            "heatmap": debug_accum / denom,
            "output_size": (out_h, out_w),
        }
    return sr, mask, debug


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
    tile_size: int = 0,
    tile_overlap: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    lr = image_to_tensor(image_path)
    lr, original_size = pad_lr_tensor_to_even(lr, multiple=LR_PAD_MULTIPLE)
    tile_size, tile_overlap = validate_tile_args(tile_size, tile_overlap)

    if tile_size > 0:
        print(f"[Infer] tiled inference: tile_size={tile_size}, tile_overlap={tile_overlap} LR pixels")
        sr, mask, debug = run_tiled_model_inference(
            model,
            lr,
            device=device,
            save_debug=save_debug,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    else:
        lr = lr.to(device)
        sr, mask, debug = _forward_model(model, lr, save_debug=save_debug)

    scale = int(getattr(model, "SR_SCALE", 4))
    sr = crop_sr_to_original(sr, original_size, scale=scale)
    mask = crop_sr_to_original(mask, original_size, scale=scale)
    if debug is not None and debug.get("heatmap") is not None:
        debug = dict(debug)
        debug["heatmap"] = crop_sr_to_original(debug["heatmap"], original_size, scale=scale)
        debug["output_size"] = tuple(sr.shape[-2:])
    elif debug is not None and debug.get("a_tex") is not None:
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
    if save_debug and debug is not None and debug.get("heatmap") is not None:
        tensor_to_pil(debug["heatmap"]).save(debug_path)
    elif save_debug and debug is not None and debug.get("a_tex") is not None:
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
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
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
    parser.add_argument("--tile-size", type=int, default=0, help="LR patch size for tiled inference; 0 disables tiling")
    parser.add_argument("--tile-overlap", type=int, default=32, help="LR overlap between neighboring patches")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run_inference(parse_args(argv))


if __name__ == "__main__":
    main()
