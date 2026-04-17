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
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from models.tadisr_model import TADiSRWrapper
from losses.composite_loss import CompositeTADiSRLoss
from data.dataset import TADiSRDataset, tadisr_collate_fn


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


def train(args):
    device = args.device
    print(f"Starting TADiSR training on {device}...")
    print(f"  LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Gradient clip: {args.grad_clip}")

    # 1. Dataset
    dataset = create_datasets(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tadisr_collate_fn,
        num_workers=0,  # For CPU testing
        drop_last=True,
    )

    # 2. Model
    model = TADiSRWrapper(
        use_kolors=args.use_kolors,
        context_dim=args.context_dim,
        base_jsd_dim=args.jsd_dim,
        lora_rank=args.lora_rank,
    ).to(device)

    # 3. Optimizer — only trainable parameters (LoRA + JSD)
    trainable_params = model.get_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in trainable_params)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {train_params:,} ({100*train_params/total_params:.2f}%)")

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
            x_pred, s_pred = model.forward_train(
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

            if global_step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"[Step {global_step:6d}] "
                    f"Loss: {loss.item():.4f} "
                    f"(Img: {loss_img.item():.4f}, "
                    f"Seg: {loss_seg.item():.4f}) "
                    f"LR: {current_lr:.2e}"
                )

            # Save checkpoint
            if args.save_every > 0 and global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"tadisr_step{global_step}.pt")
                os.makedirs(args.ckpt_dir, exist_ok=True)
                # Fix T2: Filter by parameter name, not by requires_grad on
                # detached state_dict tensors (which is always False)
                trainable_names = {id(p) for p in trainable_params}
                save_keys = set()
                for name, param in model.named_parameters():
                    if id(param) in trainable_names:
                        save_keys.add(name)
                    elif 'jsd' in name or 'taca' in name:
                        save_keys.add(name)
                model_sd = model.state_dict()
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

    print(f"\nTraining complete! {global_step} iterations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TADiSR Training")

    # Data
    parser.add_argument("--ftsr_dir", type=str, default="dataset/FTSR",
                        help="Path to FTSR dataset")
    parser.add_argument("--realce_dir", type=str, default="dataset/Real-CE",
                        help="Path to Real-CE dataset")

    # Model
    parser.add_argument("--use_kolors", action="store_true",
                        help="Use real Kolors backbone (requires GPU + downloads)")
    parser.add_argument("--context_dim", type=int, default=4096,
                        help="Text encoder context dimension (4096 for Kolors/ChatGLM)")
    parser.add_argument("--jsd_dim", type=int, default=128,
                        help="JSD base dimension")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank for UNet cross-attention fine-tuning")

    # Training
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU (paper: 1)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (paper: 5e-5)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm")
    parser.add_argument("--warmup_iters", type=int, default=0,
                        help="Learning rate warmup iterations (paper: 0, fixed LR)")
    parser.add_argument("--max_iters", type=int, default=200000,
                        help="Max training iterations (paper: 200k)")
    parser.add_argument("--epochs", type=int, default=9999,
                        help="Max epochs (controlled by max_iters)")

    # Logging / Checkpointing
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train(args)
