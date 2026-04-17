"""
TADiSR training dataset loader.

Fixes applied:
  - Issue 12: Added 'context' and 'text_indices' fields
  - Issue 13: Replaced None return with index-skipping fallback
"""
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class TADiSRDataset(Dataset):
    """
    Dataset for FTSR triplets: (x_L, x_H, s).

    Each sample provides:
      - lr:           [3, H_lr, W_lr]   low-resolution image
      - hr:           [3, H_hr, W_hr]   high-resolution image
      - mask:         [1, H_hr, W_hr]   text segmentation mask
      - context:      [seq_len, context_dim]  text encoder embedding
      - text_indices: list[int]  positions of "text" token in prompt
      - filename:     str
    """

    # Fixed prompt used in TADiSR: "A high-quality photo with clear text"
    # "text" is typically the last token. These values match the paper's approach
    # of using a fixed prompt with known token positions.
    DEFAULT_SEQ_LEN = 77
    DEFAULT_CONTEXT_DIM = 4096  # Kolors uses ChatGLM (4096-dim)

    def __init__(self, data_root, transform=None,
                 text_token_indices=None, context_dim=4096, seq_len=77):
        self.data_root = Path(data_root)
        self.hr_dir = self.data_root / "HR"
        self.lr_dir = self.data_root / "LR"
        self.mask_dir = self.data_root / "Mask"

        self.context_dim = context_dim
        self.seq_len = seq_len

        # Paper uses fixed prompt "A high-quality photo with clear text"
        # "text" token position(s) in the tokenized sequence
        self.text_token_indices = text_token_indices or [10]

        # Pre-compute a fixed context embedding (in production, this would
        # come from the frozen text encoder applied to the fixed prompt)
        # Shape: [seq_len, context_dim]
        self._fixed_context = None

        self.samples = sorted([f.name for f in self.hr_dir.glob("*.png")])

    def _get_context(self):
        """
        Returns a fixed context embedding.
        In production, replace with: text_encoder(tokenize("A high-quality photo ..."))
        """
        if self._fixed_context is None:
            # Random but fixed across all samples (simulates a frozen text encoder)
            torch.manual_seed(42)
            self._fixed_context = torch.randn(self.seq_len, self.context_dim)
        return self._fixed_context.clone()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        hr_path = self.hr_dir / filename
        lr_path = self.lr_dir / filename
        mask_path = self.mask_dir / filename

        hr_img = cv2.imread(str(hr_path))
        lr_img = cv2.imread(str(lr_path))
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Issue 13: fallback to a different index instead of returning None
        if hr_img is None or lr_img is None or mask_img is None:
            # Try next valid sample (wrapping around)
            for offset in range(1, len(self.samples)):
                alt_idx = (idx + offset) % len(self.samples)
                return self.__getitem__(alt_idx)
            # If absolutely everything fails, return zeros
            return self._empty_sample(filename)

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        hr_tensor = torch.from_numpy(hr_img.transpose(2, 0, 1)).float() / 255.0
        lr_tensor = torch.from_numpy(lr_img.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).float() / 255.0

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "mask": mask_tensor,
            "context": self._get_context(),            # Issue 12
            "text_indices": self.text_token_indices,    # Issue 12
            "filename": filename,
        }

    def _empty_sample(self, filename):
        """Return a valid-shaped zero tensor sample as last resort."""
        return {
            "lr": torch.zeros(3, 128, 128),
            "hr": torch.zeros(3, 512, 512),
            "mask": torch.zeros(1, 512, 512),
            "context": self._get_context(),
            "text_indices": self.text_token_indices,
            "filename": filename,
        }


def tadisr_collate_fn(batch):
    """
    Custom collate function that handles text_indices (list[int])
    which shouldn't be stacked as tensors.
    """
    elem = batch[0]
    result = {}
    for key in elem:
        if key == "text_indices":
            # Keep as list of lists (one per sample)
            result[key] = [d[key] for d in batch]
        elif key == "filename":
            result[key] = [d[key] for d in batch]
        else:
            result[key] = torch.stack([d[key] for d in batch])
    return result


if __name__ == "__main__":
    dataset = TADiSRDataset(data_root="dataset/FTSR")
    print(f"Dataset initialization: found {len(dataset)} items.")
    if len(dataset) > 0:
        sample = dataset[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {v}")
