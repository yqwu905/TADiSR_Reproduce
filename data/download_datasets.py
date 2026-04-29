#!/usr/bin/env python3
"""
Download script for FTSR pipeline datasets:
  1. CTR (Chinese Text Recognition) - scene subset from Google Drive
  2. LSDIR subset (high-quality backgrounds) - from HuggingFace
  3. Hi-SAM / SAM-TS model weights - for text segmentation
  4. SAM ViT-B base weights - required by Hi-SAM
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def ensure_package(pkg, pip_name=None):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or pkg])

def download_ctr(output_dir: Path):
    """Download CTR scene dataset (LMDB) from Google Drive."""
    ensure_package("gdown")
    import gdown

    output_dir.mkdir(parents=True, exist_ok=True)
    ctr_dir = output_dir / "CTR_scene"

    if ctr_dir.exists() and any(ctr_dir.iterdir()):
        print(f"[CTR] Already exists at {ctr_dir}, skipping.")
        return ctr_dir

    # Google Drive folder ID for CTR scene dataset
    # From: https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5
    gdrive_url = "https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5"

    print(f"[CTR] Downloading scene dataset from Google Drive...")
    print(f"[CTR] Target: {ctr_dir}")
    print(f"[CTR] NOTE: This is ~2GB. If gdown fails, manually download from:")
    print(f"[CTR]   {gdrive_url}")
    print(f"[CTR] and extract to {ctr_dir}")

    try:
        gdown.download_folder(gdrive_url, output=str(ctr_dir), quiet=False)
    except Exception as e:
        print(f"[CTR] Auto-download failed: {e}")
        print(f"[CTR] Please download manually from the URL above.")
        # Create a placeholder so the script can continue
        ctr_dir.mkdir(parents=True, exist_ok=True)

    return ctr_dir


def download_lsdir_subset(output_dir: Path, num_images: int = 1000):
    """Download a subset of LSDIR from HuggingFace."""
    ensure_package("huggingface_hub")
    from huggingface_hub import hf_hub_download, list_repo_tree

    lsdir_dir = output_dir / "LSDIR"
    lsdir_dir.mkdir(parents=True, exist_ok=True)

    existing = list(lsdir_dir.glob("*.png")) + list(lsdir_dir.glob("*.jpg"))
    if len(existing) >= num_images:
        print(f"[LSDIR] Already have {len(existing)} images, skipping.")
        return lsdir_dir

    print(f"[LSDIR] Downloading {num_images} images from HuggingFace ofsoundof/LSDIR...")
    print(f"[LSDIR] NOTE: LSDIR is large. We download only the first {num_images} images.")
    print(f"[LSDIR] If this fails, you can manually download from:")
    print(f"[LSDIR]   https://huggingface.co/datasets/ofsoundof/LSDIR")

    try:
        # LSDIR on HuggingFace is organized in tar files.
        # We'll download just the first shard and extract what we need.
        local_path = hf_hub_download(
            repo_id="ofsoundof/LSDIR",
            filename="train/hr/part_1.tar",
            repo_type="dataset",
            local_dir=str(output_dir / "LSDIR_cache"),
        )
        print(f"[LSDIR] Downloaded shard to {local_path}")

        # Extract subset
        import tarfile
        count = 0
        with tarfile.open(local_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, path=str(lsdir_dir))
                    count += 1
                    if count >= num_images:
                        break
        print(f"[LSDIR] Extracted {count} images to {lsdir_dir}")

    except Exception as e:
        print(f"[LSDIR] HuggingFace download failed: {e}")
        raise RuntimeError(
            "Failed to download LSDIR backgrounds from HuggingFace. "
            "Download the dataset manually from https://huggingface.co/datasets/ofsoundof/LSDIR "
            f"and place images under {lsdir_dir}."
        ) from e

    return lsdir_dir


def download_hisam_weights(output_dir: Path):
    """Download Hi-SAM (SAM-TS) weights for text segmentation."""
    weights_dir = output_dir / "pretrained_checkpoint"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # SAM-TS ViT-B weights (smallest, ~375MB total with SAM base)
    sam_ts_path = weights_dir / "sam_tss_b_hiertext.pth"
    sam_base_path = weights_dir / "sam_vit_b_01ec64.pth"

    if sam_ts_path.exists():
        print(f"[Hi-SAM] SAM-TS weights already exist at {sam_ts_path}")
    else:
        print(f"[Hi-SAM] SAM-TS weights need to be downloaded manually.")
        print(f"[Hi-SAM] Download from OneDrive:")
        print(f"[Hi-SAM]   SAM-TS ViT-B: https://1drv.ms/u/s!AimBgYV7JjTlgcoycYfJS3jn8Zi5aQ?e=qsGFu4")
        print(f"[Hi-SAM] Save to: {sam_ts_path}")

    if sam_base_path.exists():
        print(f"[Hi-SAM] SAM ViT-B base weights already exist at {sam_base_path}")
    else:
        # SAM ViT-B base weights can be downloaded directly
        sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        print(f"[Hi-SAM] Downloading SAM ViT-B base weights (~375MB)...")
        try:
            import requests
            from tqdm import tqdm
            resp = requests.get(sam_url, stream=True, timeout=60)
            total = int(resp.headers.get('content-length', 0))
            with open(sam_base_path, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True, desc="SAM ViT-B") as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"[Hi-SAM] SAM ViT-B saved to {sam_base_path}")
        except Exception as e:
            print(f"[Hi-SAM] Download failed: {e}")
            print(f"[Hi-SAM] Please download manually from: {sam_url}")

    return weights_dir


def clone_hisam_repo(output_dir: Path):
    """Clone the Hi-SAM repository for inference code."""
    hisam_dir = output_dir / "Hi-SAM"

    if hisam_dir.exists() and (hisam_dir / "demo_hisam.py").exists():
        print(f"[Hi-SAM] Repository already cloned at {hisam_dir}")
        return hisam_dir

    print(f"[Hi-SAM] Cloning Hi-SAM repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/ymy-k/Hi-SAM.git", str(hisam_dir)],
        check=True
    )
    print(f"[Hi-SAM] Cloned to {hisam_dir}")
    return hisam_dir


def setup_ctr_from_lmdb(ctr_lmdb_dir: Path, output_dir: Path, max_images: int = 5000):
    """Extract images from CTR LMDB into individual PNG files with GT label filenames."""
    ensure_package("lmdb")
    import lmdb

    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("*.png"))
    if len(existing) >= max_images:
        print(f"[CTR] Already have {len(existing)} extracted images, skipping.")
        return output_dir

    # Find the LMDB directory
    lmdb_paths = list(ctr_lmdb_dir.rglob("data.mdb"))
    if not lmdb_paths:
        print(f"[CTR] No LMDB files found in {ctr_lmdb_dir}. Please download CTR first.")
        return output_dir

    lmdb_path = lmdb_paths[0].parent
    print(f"[CTR] Extracting images from LMDB: {lmdb_path}")

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    count = 0

    with env.begin() as txn:
        num_samples = int(txn.get(b'num-samples').decode()) if txn.get(b'num-samples') else 0
        print(f"[CTR] Found {num_samples} samples in LMDB")

        for i in range(1, min(num_samples + 1, max_images + 1)):
            img_key = f'image-{i:09d}'.encode()
            label_key = f'label-{i:09d}'.encode()

            img_data = txn.get(img_key)
            label_data = txn.get(label_key)

            if img_data is None:
                continue

            label = label_data.decode('utf-8').strip() if label_data else f"unknown_{i}"
            # Sanitize filename
            safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
            fname = f"{safe_label}_{i:06d}.png"

            import numpy as np
            import cv2
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(str(output_dir / fname), img)
                count += 1

    env.close()
    print(f"[CTR] Extracted {count} images to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for FTSR pipeline")
    parser.add_argument("--data_root", type=str, default=str(ROOT / "raw_data"),
                       help="Root directory for raw data")
    parser.add_argument("--lsdir_count", type=int, default=1000,
                       help="Number of LSDIR background images to download")
    parser.add_argument("--ctr_count", type=int, default=5000,
                       help="Max CTR images to extract")
    parser.add_argument("--skip_ctr", action="store_true", help="Skip CTR download")
    parser.add_argument("--skip_lsdir", action="store_true", help="Skip LSDIR download")
    parser.add_argument("--skip_hisam", action="store_true", help="Skip Hi-SAM setup")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FTSR Pipeline - Dataset Download Script")
    print("=" * 60)

    # Step 1: CTR dataset
    if not args.skip_ctr:
        print("\n--- Step 1: CTR Dataset ---")
        ctr_dir = download_ctr(data_root)

    # Step 2: LSDIR backgrounds
    if not args.skip_lsdir:
        print("\n--- Step 2: LSDIR Background Images ---")
        download_lsdir_subset(data_root, num_images=args.lsdir_count)

    # Step 3: Hi-SAM
    if not args.skip_hisam:
        print("\n--- Step 3: Hi-SAM Setup ---")
        clone_hisam_repo(ROOT / "third_party")
        download_hisam_weights(ROOT / "third_party" / "Hi-SAM")

    print("\n" + "=" * 60)
    print("Download script complete.")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. If CTR auto-download failed, manually download LMDB from Google Drive")
    print(f"  2. If SAM-TS weights are missing, download from OneDrive (see messages above)")
    print(f"  3. Run: python data/build_ftsr.py")
