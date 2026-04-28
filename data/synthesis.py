import os
import json
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List

class TADiSRDataSynthesizer:
    def __init__(self, fg_dir, mask_dir, fg_txt, bg_txt, output_dir, target_size=(1024, 768)):
        """
        fg_dir: Directory containing foreground text patches
        mask_dir: Directory containing foreground text segmentation masks (1-channel or 3-channel binary)
        fg_txt: Json file that records OCR verification results for foregrounds
        bg_txt: Directory containing high-quality background images (text-free)
        output_dir: Where to save the synthesized HR images and merged masks
        """
        self.fg_dir = Path(fg_dir)
        self.mask_dir = Path(mask_dir)
        self.bg_txt = Path(bg_txt)
        self.fg_txt = Path(fg_txt)
        self.output_dir = Path(output_dir)
        
        self.hr_out = self.output_dir / "HR"
        self.mask_out = self.output_dir / "Mask"
        self.hr_out.mkdir(parents=True, exist_ok=True)
        self.mask_out.mkdir(parents=True, exist_ok=True)
        
        self.target_size = target_size
        
        self.bgs = self.load_backgrounds()
        self.fgs = self.load_verified_foregrounds()
        
        if len(self.bgs) == 0:
            print("Warning: No background images found.")
        if len(self.fgs) == 0:
            print("Warning: No verified foreground images found.")

    def load_verified_foregrounds(self):
        with open(self.fg_txt, 'r') as fp:
            items = [line.strip() for line in fp if line.strip()]

        fgs = []
        for name in tqdm(items, desc="Load foreground"):
            fg_path = self.fg_dir / name
            mask_path = self.mask_dir / name
            if fg_path.exists() and mask_path.exists():
                fgs.append(fg_path)
        return fgs

    def load_backgrounds(self):
        with open(self.bg_txt, 'r') as fp:
            bgs = [line.strip() for line in fp if line.strip()]
        return bgs

    def random_crop_background(self, bg):
        target_w, target_h = self.target_size
        h, w = bg.shape[:2]
        x_min = random.randint(0, w - target_w)
        y_min = random.randint(0, h - target_h)
        return bg[y_min:y_min+target_h, x_min:x_min+target_w]

    def random_transform(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1) 随机缩放
        # scale = random.uniform(0.5, 1.5)
        # h, w = img.shape[:2]
        # new_w, new_h = int(w * scale), int(h * scale)
    
        # img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
        # 2) 随机旋转（修复：扩展画布，避免旋转后裁切）
        angle = random.uniform(-15.0, 15.0)
    
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
    
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
        # 计算旋转后的外接框尺寸
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
    
        bound_w = int(np.ceil(h * sin + w * cos))
        bound_h = int(np.ceil(h * cos + w * sin))
    
        # 平移，使旋转后的内容居中放到新画布里
        M[0, 2] += (bound_w / 2.0) - cx
        M[1, 2] += (bound_h / 2.0) - cy
    
        # 图像旋转
        img = cv2.warpAffine(
            img,
            M,
            (bound_w, bound_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    
        # mask 旋转：一定要用最近邻
        mask = cv2.warpAffine(
            mask,
            M,
            (bound_w, bound_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    
        # 3) 重新二值化 mask，避免边缘异常
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
        # 4) 可选：按 mask 的有效区域裁掉多余黑边，让 patch 更紧凑
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            img = img[y_min:y_max + 1, x_min:x_max + 1]
            mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    
        return img, mask

    def composite(self, bg_img: np.ndarray, fg_patches: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Composite multiple foreground patches onto a background, producing the HR image and a combined mask.
        修改点：
        1. 新增 occupied 区域，避免多个 patch 相互重叠
        2. 每个 patch 随机尝试多个位置，找不到合法位置则跳过
        """
        bg_h, bg_w = bg_img.shape[:2]
        comp_img = bg_img.copy()
        comp_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

        # 记录已经占用的位置（基于前景有效 mask）
        occupied = np.zeros((bg_h, bg_w), dtype=bool)

        # 可调参数：每个 patch 最多尝试多少次找位置
        max_place_attempts = 50

        for (fg_img, fg_mask) in fg_patches:
            fg_img, fg_mask = self.random_transform(fg_img, fg_mask)
            fh, fw = fg_img.shape[:2]

            # 跳过太大的 patch
            if fh >= bg_h or fw >= bg_w:
                continue

            # 当前 patch 的有效区域
            fg_valid = (fg_mask > 0)
            if not np.any(fg_valid):
                continue

            placed = False

            for _ in range(max_place_attempts):
                x_min = random.randint(0, bg_w - fw)
                y_min = random.randint(0, bg_h - fh)

                # 取出候选位置的已占用区域
                occ_roi = occupied[y_min:y_min + fh, x_min:x_min + fw]

                # 如果新 patch 的有效区域与已占用区域有交集，则不能放
                if np.any(occ_roi & fg_valid):
                    continue

                # 可以放置，执行贴图
                alpha = fg_valid.astype(np.float32)
                if alpha.ndim == 2:
                    alpha = np.expand_dims(alpha, axis=-1)

                roi_img = comp_img[y_min:y_min + fh, x_min:x_min + fw]
                blended = roi_img * (1 - alpha) + fg_img * alpha
                comp_img[y_min:y_min + fh, x_min:x_min + fw] = blended.astype(np.uint8)

                # 更新总 mask
                roi_mask = comp_mask[y_min:y_min + fh, x_min:x_min + fw]
                roi_mask = np.maximum(roi_mask, fg_mask)
                comp_mask[y_min:y_min + fh, x_min:x_min + fw] = roi_mask

                # 更新占用区域
                occupied[y_min:y_min + fh, x_min:x_min + fw] |= fg_valid

                placed = True
                break

            # 如果 placed=False，说明尝试多次都没找到不重叠位置，直接跳过该 patch
            # 这里不报错，保持原有流程稳健
            if not placed:
                continue

        return comp_img, comp_mask

    def generate_samples(self, n_samples):
        print(f"Generating {n_samples} composite samples...")
        for i in tqdm(range(n_samples)):
            if len(self.bgs) == 0 or len(self.fgs) == 0:
                print("Missing data to composite.")
                break
                
            bg_path = random.choice(self.bgs)
            bg = cv2.imread(str(bg_path))
            if bg is None: continue
            
            bg = self.random_crop_background(bg)
            
            # Select 1 ~ 5 foreground patches
            num_fgs = random.randint(1, 5)
            patches = []
            for _ in range(num_fgs):
                fg_path = random.choice(self.fgs)
                fg = cv2.imread(str(fg_path))
                
                # Fetch corresponding mask
                mask_path = self.mask_dir / fg_path.name
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if fg is not None and mask is not None:
                    # Convert mask to 0/255 if not already
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    patches.append((fg, mask))
                    
            hr_img, hr_mask = self.composite(bg, patches)
            
            filename = f"synth_{i:06d}.png"
            cv2.imwrite(str(self.hr_out / filename), hr_img)
            cv2.imwrite(str(self.mask_out / filename), hr_mask)

def parse_target_size(target_size):
    try:
        w, h = target_size.lower().split("x")
        return int(w), int(h)
    except Exception as e:
        raise ValueError("target_size must be in WxH format, e.g. 1024x768") from e

if __name__ == "__main__":
    # Example Offline Generation Pipeline Runner
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_dir", type=str, default="raw_data/verified_foregrounds", help="Path to text foregrounds")
    parser.add_argument("--mask_dir", type=str, default="raw_data/verified_masks", help="Path to text masks")
    parser.add_argument("--fg_txt", type=str, required=True, help="Path to OCR verification json")
    parser.add_argument("--bg_txt", type=str, default="raw_data/backgrounds", help="Path to text-free backgrounds")
    parser.add_argument("--output_dir", type=str, default="dataset/FTSR", help="Output dir for FTSR dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of HR samples to compose")
    parser.add_argument("--target_size", type=str, default="1024x768", help="Target size in WxH format")
    args = parser.parse_args()
    
    synth = TADiSRDataSynthesizer(
        fg_dir=args.fg_dir,
        mask_dir=args.mask_dir,
        fg_txt=args.fg_txt,
        bg_txt=args.bg_txt,
        output_dir=args.output_dir,
        target_size=parse_target_size(args.target_size)
    )
    synth.generate_samples(args.num_samples)
