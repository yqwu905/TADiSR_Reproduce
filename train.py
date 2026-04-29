"""
TADiSR training script.

Paper reference (Section 4.1):
  - Kolors variant of LDM
  - Fixed diffusion timestep t = 200
  - AdamW optimizer, lr = 5e-5
  - 4× H20 GPUs, batch_size = 1/GPU, 200k iterations
  - 4× super-resolution
  - Training data: FTSR (45k) + Real-CE (337 pairs)
  - Total loss: ℓ_tot = ℓ_img + ℓ_seg (NO separate noise prediction loss)
"""
import os
import yaml
import argparse
from contextlib import nullcontext
from types import SimpleNamespace
import torch
import torch.optim as optim
import torch.distributed as dist
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


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


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
    cfg.setdefault("gradient_checkpointing", True)
    cfg.setdefault("fsdp_auto_wrap", True)
    cfg.setdefault("fsdp_activation_checkpointing", True)
    cfg.setdefault("lpips_resize", 512)
    cfg.setdefault("taca_query_chunk_size", 1024)
    cfg.setdefault("taca_checkpoint", True)
    cfg.setdefault("taca_detach", False)
    cfg.setdefault("fail_on_lora_error", True)
    cfg.setdefault("require_precomputed_text_context", False)

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
        print("[Data] No real data found. Using dummy dataset for testing.")
        return _create_dummy_dataset(args)

    if len(datasets) > 1:
        return ConcatDataset(datasets)
    return datasets[0]


def _create_dummy_dataset(args):
    """Dummy dataset for local CPU testing."""
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return {
                "lr": torch.rand(3, 128, 128),
                "hr": torch.rand(3, 512, 512),
                "mask": torch.rand(1, 512, 512),
                "context": torch.rand(77, args.context_dim),
                "text_indices": [5],
                "filename": f"dummy_{idx:04d}.png",
            }
    return DummyDataset()


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
    ddp_state = setup_distributed(args)
    is_main = ddp_state["rank"] == 0
    device = ddp_state["device"]
    dist_strategy = str(getattr(args, "dist_strategy", "ddp")).lower()
    precision_name, precision_dtype = _resolve_precision_dtype(getattr(args, "precision", "bf16"))
    if dist_strategy not in {"ddp", "fsdp"}:
        raise ValueError(f"Unsupported dist_strategy: {dist_strategy}. Use 'ddp' or 'fsdp'.")

    if is_main:
        print(f"Starting TADiSR training on {device}...")
        print(f"  LR: {args.lr}, Batch size(per-process): {args.batch_size}")
        print(f"  Max iterations: {args.max_iters}")
        print(f"  Gradient clip: {args.grad_clip}")
        print(f"  Precision: {precision_name}")
        print(f"  Gradient checkpointing: {bool(getattr(args, 'gradient_checkpointing', False))}")
        print(f"  TACA chunk: {getattr(args, 'taca_query_chunk_size', 0)}")
        print(f"  TACA checkpoint: {bool(getattr(args, 'taca_checkpoint', False))}")
        print(f"  LPIPS resize: {getattr(args, 'lpips_resize', 0)}")
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
        num_workers=0,  # For CPU testing
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

    # 4. Learning rate scheduler (optional warmup + cosine)
    if args.warmup_iters > 0:
        def lr_lambda(step):
            if step < args.warmup_iters:
                return step / max(args.warmup_iters, 1)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # 5. Loss
    criterion = CompositeTADiSRLoss(
        lpips_resize=getattr(args, "lpips_resize", 0),
    ).to(device)

    # 6. Training Loop
    model.train()
    global_step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for ep in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)

        for step, batch in enumerate(loader):
            if global_step >= args.max_iters:
                break

            lr_img = batch["lr"].to(device)
            hr_img = batch["hr"].to(device)
            mask = batch["mask"].to(device)
            context = batch["context"].to(device)
            text_indices = batch["text_indices"][0]

            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, precision_name, precision_dtype):
                # Forward — Paper loss: ℓ_tot = ℓ_img + ℓ_seg (NO noise loss)
                x_pred, s_pred = model.forward_train(
                    lr_img, context, text_indices
                )

                # Calculate composite loss
                loss, loss_img, loss_seg = criterion(x_pred, hr_img, s_pred, mask)
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

            if is_main and global_step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalar("train/loss", loss.item(), global_step)
                tb_writer.add_scalar("train/loss_img", loss_img.item(), global_step)
                tb_writer.add_scalar("train/loss_seg", loss_seg.item(), global_step)
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
                os.makedirs(args.ckpt_dir, exist_ok=True)
                # Fix T2: Filter by parameter name, not by requires_grad on
                # detached state_dict tensors (which is always False)
                trainable_names = {id(p) for p in trainable_params}
                save_keys = set()
                for name, param in raw_model.named_parameters():
                    if id(param) in trainable_names:
                        save_keys.add(name)
                    elif 'jsd' in name or 'taca' in name:
                        save_keys.add(name)
                model_sd = raw_model.state_dict()
                save_sd = {k: v for k, v in model_sd.items() if k in save_keys}
                torch.save({
                    'step': global_step,
                    'model_state_dict': save_sd,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, ckpt_path)
                print(f"  Checkpoint saved to {ckpt_path} ({len(save_sd)} tensors)")

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

    cli_args = parser.parse_args()
    args = load_config(cli_args.config)
    train(args)
