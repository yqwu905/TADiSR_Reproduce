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
from types import SimpleNamespace
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from models.tadisr_model import TADiSRWrapper
from losses.composite_loss import CompositeTADiSRLoss
from data.dataset import TADiSRDataset, tadisr_collate_fn



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

    if is_main:
        print(f"Starting TADiSR training on {device}...")
        print(f"  LR: {args.lr}, Batch size(per-process): {args.batch_size}")
        print(f"  Max iterations: {args.max_iters}")
        print(f"  Gradient clip: {args.grad_clip}")
        if ddp_state["distributed"]:
            print(f"  DDP enabled — world_size: {ddp_state['world_size']}")

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
    ).to(device)

    # 3. Optimizer — only trainable parameters (LoRA + JSD)
    trainable_params = model.get_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in trainable_params)
    if is_main:
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {train_params:,} ({100*train_params/total_params:.2f}%)")

    if ddp_state["distributed"]:
        model = DDP(
            model,
            device_ids=[ddp_state["local_rank"]] if device.type in {"cuda", "npu"} else None,
            output_device=ddp_state["local_rank"] if device.type in {"cuda", "npu"} else None,
            find_unused_parameters=False,
        )
    raw_model = model.module if ddp_state["distributed"] else model

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

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
    criterion = CompositeTADiSRLoss().to(device)

    # 6. Training Loop
    model.train()
    global_step = 0

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

            optimizer.zero_grad()

            # Forward — Paper loss: ℓ_tot = ℓ_img + ℓ_seg (NO noise loss)
            x_pred, s_pred = raw_model.forward_train(
                lr_img, context, text_indices
            )

            # Calculate composite loss
            loss, loss_img, loss_seg = criterion(x_pred, hr_img, s_pred, mask)

            # Backward with gradient clipping
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            global_step += 1

            if is_main and global_step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Step {global_step:6d}] "
                    f"Loss: {loss.item():.4f} "
                    f"(Img: {loss_img.item():.4f}, "
                    f"Seg: {loss_seg.item():.4f}) "
                    f"LR: {current_lr:.2e}"
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
