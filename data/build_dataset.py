"""
Full dataset build pipeline that runs Synthesis → Degradation → Validation.

Fix applied:
  - Issue 18: Fixed relative imports to absolute package imports
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.synthesis import TADiSRDataSynthesizer
from data.degradation import run_degradation_pipeline
from data.dataset import TADiSRDataset


def build_dataset(args):
    dataset_dir = Path(args.output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("--- Step 1/3: Data Synthesis ---")
    synth = TADiSRDataSynthesizer(
        fg_dir=args.fg_dir,
        mask_dir=args.mask_dir,
        bg_dir=args.bg_dir,
        output_dir=dataset_dir
    )
    synth.generate_samples(args.num_samples)
    print("Data synthesis complete!\n")

    print("--- Step 2/3: HR -> LR Degradation ---")
    run_degradation_pipeline(
        hr_dir=dataset_dir / "HR",
        lr_dir=dataset_dir / "LR",
        sf=args.sf
    )
    print("Data degradation complete!\n")

    print("--- Step 3/3: Validation via DataLoader ---")
    dataset = TADiSRDataset(data_root=dataset_dir)
    print(f"Dataset successfully initialized. Found {len(dataset)} items.")
    if len(dataset) > 0:
        sample = dataset[0]
        if sample is not None:
            print("Sample Data Check:")
            for k, v in sample.items():
                if hasattr(v, "shape"):
                    print(f"  - {k}: {v.shape}")
                else:
                    print(f"  - {k}: {v}")
        else:
            print("Warning: First sample failed to load (returned None).")

    print("\n--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full build dataset pipeline for TADiSR."
    )
    parser.add_argument("--fg_dir", type=str,
                        default="raw_data/verified_foregrounds",
                        help="Path to text foregrounds")
    parser.add_argument("--mask_dir", type=str,
                        default="raw_data/verified_masks",
                        help="Path to text masks")
    parser.add_argument("--bg_dir", type=str,
                        default="raw_data/backgrounds",
                        help="Path to background images")
    parser.add_argument("--output_dir", type=str,
                        default="dataset/FTSR",
                        help="Output dir for final dataset")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate")
    parser.add_argument("--sf", type=int, default=4,
                        help="Super-resolution scale factor")

    args = parser.parse_args()
    build_dataset(args)
