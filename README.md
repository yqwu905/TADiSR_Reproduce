# TADiSR Reproduce

**Paper**: *Text-Aware Real-World Image Super-Resolution via Diffusion Model with Joint Segmentation Decoders*
**arXiv**: [2506.04641](https://arxiv.org/abs/2506.04641)

本仓库为 TADiSR 论文的非官方复现，包含完整的数据合成管线、模型架构和训练代码。

---

## 目录

- [1. 环境配置](#1-环境配置)
- [2. 项目结构](#2-项目结构)
- [3. 预训练模型权重下载](#3-预训练模型权重下载)
- [4. 数据集准备](#4-数据集准备)
- [5. FTSR 数据构造管线](#5-ftsr-数据构造管线)
- [6. 训练](#6-训练)
- [7. 模型架构说明](#7-模型架构说明)
- [8. 损失函数说明](#8-损失函数说明)
- [9. 常见问题](#9-常见问题)

---

## 1. 环境配置

### 1.1 基础依赖

```bash
# 创建虚拟环境
conda create -n tadisr python=3.10
conda activate tadisr

# 安装 PyTorch（CUDA 11.8 示例）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

### 1.2 可选依赖（按需安装）

| 依赖 | 用途 | 安装命令 |
|------|------|----------|
| `lpips` | 感知损失（论文必需） | `pip install lpips` |
| `paddlepaddle` + `paddleocr` | PP-OCR 中文识别（论文选择） | `pip install paddlepaddle paddleocr` |
| `scipy` | 广义高斯核降质 | `pip install scipy` |
| `gdown` | Google Drive 下载 CTR 数据集 | `pip install gdown` |
| `huggingface_hub` | 下载 LSDIR 数据集 | `pip install huggingface_hub` |

### 1.3 硬件要求

| 配置 | 说明 |
|------|------|
| **论文配置** | 4× NVIDIA H20 GPU (96GB)，batch_size=1/GPU |
| **最低配置** | 1× NVIDIA A100/A800 (80GB)，加载完整 Kolors 骨干约需 ~40GB |
| **调试模式** | CPU 可运行（使用轻量级 fallback 组件，不加载 Kolors） |

---

## 2. 项目结构

```
TADiSR_Reproduce/
├── models/
│   ├── tadisr_model.py       # 主模型封装：VAE + UNet + TACA + JSD
│   ├── vae_jsd.py            # Joint Segmentation Decoders (JSD) + CDIB
│   └── kolors_unet_mod.py    # 轻量级 TACA UNet（CPU 调试用）
├── losses/
│   ├── __init__.py
│   └── composite_loss.py     # 复合损失：MFL + LPIPS + MSE + Focal + Dice
├── data/
│   ├── download_datasets.py  # 数据集下载脚本
│   ├── build_ftsr.py         # FTSR 数据集构造主管线
│   ├── build_dataset.py      # 简化版构造管线
│   ├── text_segmentation.py  # Hi-SAM 文本分割
│   ├── ocr_filter.py         # OCR 一致性过滤
│   ├── degradation.py        # Real-ESRGAN 两阶段降质
│   ├── synthesis.py          # 图像合成
│   └── dataset.py            # PyTorch Dataset
├── dataset/FTSR/             # 示例数据（20 组 LR/HR/Mask）
├── tests/
│   └── test_all.py           # 单元测试
├── train.py                  # 训练脚本
├── requirements.txt
└── README.md
```

---

## 3. 预训练模型权重下载

TADiSR 基于 [Kolors](https://github.com/Kwai-Kolors/Kolors) 扩散模型。训练前需要下载以下权重：

### 3.1 Kolors 骨干模型（必需）

Kolors 模型通过 HuggingFace `diffusers` 自动下载，首次运行时会自动缓存到 `~/.cache/huggingface/`。

**模型 ID**: `Kwai-Kolors/Kolors-diffusers`

包含三个组件：

| 组件 | 子目录 | 大小 | 说明 |
|------|--------|------|------|
| **VAE** (AutoencoderKL) | `vae/` | ~335MB | 编码器冻结，解码器权重用于初始化 JSD 图像解码器 |
| **UNet** (UNet2DConditionModel) | `unet/` | ~5.5GB | LoRA 微调 cross-attention 层 |
| **ChatGLM Text Encoder** | `text_encoder/` | ~12GB | 完全冻结，编码固定提示词 |

**手动预下载（推荐，避免训练时等待）**：

```bash
# 方法 1：使用 huggingface-cli
huggingface-cli download Kwai-Kolors/Kolors-diffusers --local-dir ./pretrained/Kolors-diffusers

# 方法 2：使用 Python
python -c "
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, AutoModel

print('下载 VAE...')
AutoencoderKL.from_pretrained('Kwai-Kolors/Kolors-diffusers', subfolder='vae')

print('下载 UNet...')
UNet2DConditionModel.from_pretrained('Kwai-Kolors/Kolors-diffusers', subfolder='unet')

print('下载 ChatGLM Text Encoder...')
AutoTokenizer.from_pretrained('Kwai-Kolors/Kolors-diffusers', subfolder='text_encoder', trust_remote_code=True)
AutoModel.from_pretrained('Kwai-Kolors/Kolors-diffusers', subfolder='text_encoder', trust_remote_code=True)

print('全部下载完成!')
"
```

> **注意**: ChatGLM 需要 `trust_remote_code=True`，因为它使用自定义模型代码。

### 3.2 Hi-SAM 文本分割权重（数据构造用）

Hi-SAM (SAM-TS) 用于从 CTR 数据集中生成文本分割 mask。

```bash
# 1. 克隆 Hi-SAM 仓库
mkdir -p third_party
git clone https://github.com/ymy-k/Hi-SAM.git third_party/Hi-SAM

# 2. 下载 SAM ViT-B 基础权重（~375MB）
wget -P third_party/Hi-SAM/pretrained_checkpoint/ \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 3. 下载 SAM-TS (Hi-SAM) 文本分割权重（手动下载）
# 下载地址（OneDrive）:
#   ViT-B: https://1drv.ms/u/s!AimBgYV7JjTlgcoycYfJS3jn8Zi5aQ?e=qsGFu4
#   ViT-L: https://1drv.ms/u/s!AimBgYV7JjTlgcozYjH8mFf8I01URA?e=sf0rMi
#   ViT-H: https://1drv.ms/u/s!AimBgYV7JjTlgco0E5rYpqCmkffZ2A?e=NoBGb6
# 下载后放到:
mv sam_tss_b_hiertext.pth third_party/Hi-SAM/pretrained_checkpoint/
```

### 3.3 HAT 超分辨率权重（数据构造用，可选）

论文使用 HAT 对小尺寸文字补丁进行超分辨率增强。不安装则自动 fallback 到 bicubic 上采样。

```bash
# 1. 克隆 HAT 仓库
git clone https://github.com/XPixelGroup/HAT.git third_party/HAT

# 2. 下载 HAT-SRx4 权重（~90MB）
# 从 https://github.com/XPixelGroup/HAT/releases 下载 HAT_SRx4.pth
mkdir -p third_party/HAT/pretrained
mv HAT_SRx4.pth third_party/HAT/pretrained/
```

### 3.4 LPIPS 权重（自动下载）

`lpips` 库首次使用时会自动下载 VGG 权重（~60MB），无需手动操作。

### 3.5 权重下载汇总

| 权重 | 大小 | 必需性 | 用途 |
|------|------|--------|------|
| Kolors VAE | ~335MB | **训练必需** | VAE 编码 + JSD 初始化 |
| Kolors UNet | ~5.5GB | **训练必需** | 噪声预测 + LoRA 微调 |
| Kolors ChatGLM | ~12GB | **训练必需** | 文本编码（冻结） |
| SAM ViT-B 基础权重 | ~375MB | 数据构造需要 | Hi-SAM 运行基础 |
| SAM-TS (Hi-SAM) 权重 | ~375MB | 数据构造需要 | 文本分割 mask 生成 |
| HAT-SRx4 | ~90MB | 可选 | 文字补丁超分辨率 |
| LPIPS VGG | ~60MB | 自动下载 | 感知损失 |

---

## 4. 数据集准备

### 4.1 数据集一览

论文使用以下数据集进行训练：

| 数据集 | 用途 | 样本数 | 来源 |
|--------|------|--------|------|
| **FTSR** (合成) | 主训练集 | ~45,000 | 由 CTR + LSDIR 合成 |
| **Real-CE** | 真实文本图像训练 | 337 对 | [Real-CE 数据集](https://github.com/CalcuLuUus/Real-CE) |
| **CTR** | 文字前景提取源 | ~500,000 | [CTR 中文文本识别](https://github.com/fudan-ocr) |
| **LSDIR** | 无文字背景 | ~1,000+ | [LSDIR HuggingFace](https://huggingface.co/datasets/ofsoundof/LSDIR) |

### 4.2 自动下载

```bash
# 下载所有原始数据集 + 设置 Hi-SAM
python data/download_datasets.py \
  --data_root raw_data \
  --lsdir_count 1000 \
  --ctr_count 5000

# 可选：跳过某些步骤
python data/download_datasets.py --skip_ctr      # 跳过 CTR 下载
python data/download_datasets.py --skip_lsdir    # 跳过 LSDIR 下载
python data/download_datasets.py --skip_hisam    # 跳过 Hi-SAM 克隆
```

**执行后目录结构**：

```
raw_data/
├── CTR_scene/          # CTR 数据集 LMDB
├── LSDIR/              # LSDIR 高质量背景图
├── foregrounds/        # 从 CTR 提取的文字前景（by download_datasets.py）
├── masks/              # 对应的分割 mask
├── verified_foregrounds/  # OCR 验证通过的前景
└── verified_masks/        # OCR 验证通过的 mask
```

### 4.3 手动下载 CTR（如自动下载失败）

CTR 数据集托管在 Google Drive：

1. 访问 [CTR Google Drive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5)
2. 下载 `scene` 子文件夹（LMDB 格式）
3. 解压到 `raw_data/CTR_scene/`

从 LMDB 提取单张图像：

```bash
python data/download_datasets.py --skip_lsdir --skip_hisam --ctr_count 5000
```

### 4.4 Real-CE 数据集（可选）

```bash
# 手动下载 Real-CE 并按如下结构放置
dataset/Real-CE/
├── HR/       # 高分辨率图像
├── LR/       # 低分辨率图像
└── Mask/     # 文本分割 mask（如无可跳过）
```

---

## 5. FTSR 数据构造管线

FTSR 数据集的构造严格遵循论文 Section 3.5 的流程：

### 5.1 完整管线流程

```
CTR 文字图像 ──→ Hi-SAM 文本分割 ──→ OCR 一致性过滤 ──→ 长边/字符比过滤
                                                              │
                                                              ▼
                                                        HAT 超分辨率
                                                        （小补丁增强）
                                                              │
LSDIR 背景 ──→ OCR 过滤（排除含文字背景）──→ 合成 HR 图像 ←──┘
                                                │
                                                ▼
                                     Real-ESRGAN 两阶段降质
                                                │
                                                ▼
                                       (x_L, x_H, s) 三元组
```

### 5.2 运行完整构造管线

```bash
# 完整管线（推荐用 GPU 加速 Hi-SAM 和 OCR）
python data/build_ftsr.py \
  --fg_dir raw_data/foregrounds \
  --bg_dir raw_data/LSDIR \
  --output_dir dataset/FTSR \
  --num_samples 45000 \
  --canvas_size 512 \
  --sf 4 \
  --device cuda

# 快速调试（跳过耗时步骤）
python data/build_ftsr.py \
  --num_samples 100 \
  --skip_ocr \
  --skip_hat \
  --skip_bg_ocr \
  --device cpu
```

### 5.3 管线各步骤详解

#### 步骤 1：文本分割（Hi-SAM / SAM-TS）

从 CTR 文字前景图像中提取像素级文本分割 mask：

```python
from data.text_segmentation import TextSegmentor

segmentor = TextSegmentor(device="cuda")       # 使用 Hi-SAM
mask = segmentor.segment(image_bgr)            # 返回 0/255 二值 mask
```

- 如果 Hi-SAM 未安装，自动回退到基于 Sauvola 局部自适应阈值的经典分割方法
- 输出 mask 中：0 = 背景，255 = 文字

#### 步骤 2：OCR 一致性过滤

论文核心过滤步骤 - 比较原图和 mask 后图像的 OCR 识别结果：

```python
from data.ocr_filter import OCRFilter

ocr = OCRFilter(device="cuda", lang="ch")      # 使用 PP-OCR（中文）
accepted, text_orig, text_masked, score = ocr.verify_mask(
    original_bgr, mask, gt_label="标签文本"
)
# score > 0.8 则接受该 mask
```

- 优先使用 PP-OCR（支持中文，论文选择）
- 若未安装 PaddleOCR，回退到 TrOCR（仅英文）

#### 步骤 3：长边/字符数比过滤

```python
# 过滤条件：long_edge / char_count >= min_ratio
# 默认 min_ratio = 2.0
```

移除文字过于密集的补丁，确保文字可辨识。

#### 步骤 4：HAT 超分辨率（可选）

对长边 < 128px 的小尺寸文字补丁执行 4× 超分辨率：

```python
# 自动检测小补丁并增强
# 若 HAT 模型未安装，回退到 bicubic 4× 上采样
```

#### 步骤 5：LSDIR 背景 OCR 过滤

排除包含文字的 LSDIR 背景图像，避免标注歧义：

```python
backgrounds = filter_backgrounds_ocr(backgrounds, ocr_filter)
```

#### 步骤 6：HR 图像合成

将经过滤的文字前景补丁粘贴到无文字背景上：

- 每张图像随机粘贴 1~5 个文字补丁
- 随机缩放、随机位置
- Alpha 混合确保自然过渡
- 同步生成对应的分割 mask

#### 步骤 7：Real-ESRGAN 两阶段降质

```python
from data.degradation import RealESRGANDegradation

degrader = RealESRGANDegradation(scale_factor=4)
lr_image = degrader.degrade(hr_image)
```

两阶段级联降质流程：

```
Stage 1: 模糊(4种核) → 随机缩放 → 噪声(高斯/泊松) → JPEG 压缩
                                      ↓
Stage 2: 模糊(4种核) → 随机缩放 → 噪声(高斯/泊松) → JPEG 压缩
                                      ↓
Final:   sinc 滤波 → 缩放到目标 LR 尺寸 → 可选 JPEG 压缩
```

支持的模糊核类型（匹配 Real-ESRGAN）：
- 各向同性高斯 (45%)
- 旋转各向异性高斯 (25%)
- 广义高斯 (15%)
- 平台核 (15%)

### 5.4 简化版构造管线

如果你已经有分割好的前景和 mask，可以使用简化版：

```bash
python data/build_dataset.py \
  --fg_dir raw_data/verified_foregrounds \
  --mask_dir raw_data/verified_masks \
  --bg_dir raw_data/backgrounds \
  --output_dir dataset/FTSR \
  --num_samples 1000 \
  --sf 4
```

### 5.5 构造完成后的数据结构

```
dataset/FTSR/
├── HR/                # 高分辨率图像（512×512）
│   ├── ftsr_000000.png
│   ├── ftsr_000001.png
│   └── ...
├── LR/                # 低分辨率图像（128×128）
│   ├── ftsr_000000.png
│   ├── ftsr_000001.png
│   └── ...
└── Mask/              # 文本分割 mask（512×512，单通道）
    ├── ftsr_000000.png
    ├── ftsr_000001.png
    └── ...
```

---

## 6. 训练

### 6.1 训练概述

| 参数 | 论文值 | 说明 |
|------|--------|------|
| 优化器 | AdamW | weight_decay=0.01 |
| 学习率 | 5e-5 | 固定 LR（论文未提及 scheduler） |
| 批量大小 | 1/GPU | 显存限制 |
| 总迭代数 | 200,000 | |
| 扩散时间步 | t=200（固定） | 非遍历所有时间步 |
| 固定提示词 | "A high-quality photo with clear text" | ChatGLM 编码 |
| 梯度裁剪 | 1.0 | |
| LoRA rank | 16 | 仅 cross-attention (attn2) |
| 超分辨率倍率 | 4× | |

训练参数统一放在 `YAML` 配置文件中（默认：`configs/train/default.yaml`），通过 `--config` 指定。

示例：

```bash
python train.py --config configs/train/default.yaml
```

### 6.2 完整 Kolors 训练（GPU）

先复制默认配置并修改为 GPU/Kolors 参数：

```bash
cp configs/train/default.yaml configs/train/kolors_gpu.yaml
```

将 `configs/train/kolors_gpu.yaml` 中至少以下字段改为：

```yaml
use_kolors: true
device: cuda
dist_strategy: ddp
batch_size: 1
lr: 5e-5
max_iters: 200000
log_every: 100
save_every: 5000
```

启动训练：

```bash
python train.py --config configs/train/kolors_gpu.yaml
```

> 代码中 `pretrained_path` 默认为 `"Kwai-Kolors/Kolors-diffusers"`，会自动从 HuggingFace 下载。如需修改，可在 `models/tadisr_model.py` 中调整 `TADiSRWrapper.__init__` 的 `pretrained_path` 参数。

#### 6.2.1 离线 Prompt Embedding（不加载 ChatGLM）

如果你使用固定提示词（默认 `"A high-quality photo with clear text"`），可以提前离线编码并在训练时直接加载，避免训练阶段加载 `text_encoder/`。

1) 先离线导出 embedding：

```bash
python scripts/precompute_prompt_embedding.py \
  --pretrained_path Kwai-Kolors/Kolors-diffusers \
  --prompt "A high-quality photo with clear text" \
  --output assets/fixed_prompt_context.pt
```

2) 在训练配置（如 `configs/train/kolors_gpu.yaml`）添加：

```yaml
precomputed_text_context_path: assets/fixed_prompt_context.pt
```

此时模型会读取离线文件中的：
- `context`：形状 `[1, S, C]`（或 `[S, C]`）
- `text_token_indices`：`"text"` token 的位置列表

并跳过 ChatGLM 在线编码。

### 6.3 CPU 调试模式

不加载 Kolors 骨干，使用轻量级替代组件：

```bash
cp configs/train/default.yaml configs/train/cpu_debug.yaml
# 按需修改 cpu_debug.yaml，例如 batch_size/max_iters/save_every
python train.py --config configs/train/cpu_debug.yaml
```

此模式下：
- VAE 编码器 → 4 层卷积 lightweight stub
- UNet → 2 层 down/up + skip + TACA 的轻量 UNet
- 文本上下文 → 随机固定张量
- JSD → 使用 `[32, 64, 128, 128]` 小通道配置

### 6.4 多 GPU 分布式训练

使用 PyTorch `torchrun` 进行分布式训练：

```bash
torchrun --nproc_per_node=4 train.py --config configs/train/kolors_gpu.yaml
```

可通过配置项 `dist_strategy` 切换并行策略：

```yaml
dist_strategy: ddp   # 默认
# dist_strategy: fsdp
```

> `fsdp` 需要加速器设备（`device: cuda` 或 `device: npu`）；NPU 场景依赖 `torch_npu`/环境对 FSDP 的支持。

### 6.5 训练流程详解

每个训练 step 的完整流程：

```
1. 加载 (x_L, x_H, s) 三元组
2. 冻结文本编码器：ChatGLM 编码固定提示词 → c_y
   - 从 c_y 中提取 "text" token 的位置 → text_token_indices
3. 上采样 LR：x_L ──(bicubic 4×)──→ x_L↑ (与 HR 同分辨率)
4. VAE 编码（冻结）：x_L↑ → z_L (潜空间)
5. 加噪：z_L → z_{L,t} (在固定 t=200 处加噪)
6. UNet 前向（LoRA 微调 cross-attention）：
   - 输入：z_{L,t}, t=200, c_y
   - 输出：噪声预测 n̂
   - 副产物：从所有 M 层 cross-attention 提取 "text" token 注意力图 → a_tex_raw
7. TACA 注意力聚合：
   - a_tex_raw = Concat(所有层的 "text" 注意力切片)
   - a_tex = W_a · a_tex_raw (线性投影到潜空间维度)
8. 一步去噪（论文公式 1）：
   - ẑ_H = (1/√ᾱ_t) · z_{L,t} − (√(1-ᾱ_t)/√ᾱ_t) · n̂
9. JSD 解码：(ẑ_H, a_tex) → (x̂_H, ŝ)
   - 图像解码器：ẑ_H → x̂_H (LoRA 微调，初始化自 VAE 解码器)
   - 分割解码器：a_tex → ŝ (随机初始化)
   - CDIB 交叉交互：每个分辨率级别
10. 计算损失：ℓ_tot = ℓ_img + ℓ_seg
11. 反向传播（仅更新可训练参数）
```

### 6.6 可训练参数

| 模块 | 说明 | 初始化 |
|------|------|--------|
| UNet Cross-Attention LoRA | `attn2.to_q/to_k/to_v/to_out` 的 LoRA 适配器 | 零初始化 |
| TACA W_a 投影 | 线性投影层 | 随机初始化 |
| JSD 图像解码器 LoRA | Conv2d 的 LoRA 适配器 | 零初始化 |
| JSD 分割解码器 | 完整解码器所有参数 | 随机初始化 |
| JSD CDIB | 所有交叉解码器交互块 | 零初始化（输出投影） |

**冻结参数**：VAE 编码器、ChatGLM 文本编码器、UNet 主干（非 LoRA 部分）、JSD 图像解码器基础权重。

### 6.7 Checkpoint 保存

训练过程中每 `--save_every` 步保存一次，仅保存可训练参数：

```
checkpoints/
├── tadisr_step5000.pt
├── tadisr_step10000.pt
└── ...
```

Checkpoint 内容：

```python
{
    'step': global_step,
    'model_state_dict': {...},    # 仅 LoRA + JSD + TACA 参数
    'optimizer_state_dict': {...},
    'loss': loss_value,
}
```

---

## 7. 模型架构说明

### 7.1 整体架构

```
x_L (LR) ──→ Bicubic 4× ──→ VAE Encoder (冻结) ──→ z_L
                                                       │
                                                  加噪 t=200
                                                       │
                                                       ▼
"A high-quality photo     ChatGLM      UNet + LoRA ←── z_{L,t}
 with clear text"     ──→ (冻结) ──→   (cross-attn)
                           c_y              │
                                            ├──→ 噪声预测 n̂
                                            └──→ TACA: a_tex
                                                       │
                                         一步去噪 ←───┘
                                            │
                                            ▼
                                           ẑ_H ──→ JSD 图像解码器 ──→ x̂_H (HR)
                                           a_tex ──→ JSD 分割解码器 ──→ ŝ (Mask)
                                                       ↕ CDIB 交互
```

### 7.2 JSD 解码器结构

JSD 图像解码器严格镜像 Kolors VAE 解码器架构：

```
block_out_channels = [128, 256, 512, 512]
layers_per_block   = 2 → 每 block 3 个 ResNet

解码器路径 (反转通道 [512, 512, 256, 128]):
  conv_in:       Conv2d(4, 512)
  mid_block:     ResNet(512) → Attention(512) → ResNet(512)
  ──── CDIB[0] ────
  up_block_0:    3×ResNet(512→512) + Upsample    (2×)
  ──── CDIB[1] ────
  up_block_1:    3×ResNet(512→512) + Upsample    (2×)
  ──── CDIB[2] ────
  up_block_2:    3×ResNet(512→256) + Upsample    (2×)
  ──── CDIB[3] ────
  up_block_3:    3×ResNet(256→128)               (无 upsample)
  ──── CDIB[4] ────
  conv_norm_out: GroupNorm(32, 128) → SiLU
  conv_out:      Conv2d(128, 3)  [图像] / Conv2d(128, 1)  [分割]
```

### 7.3 CDIB 交叉解码器交互

```
f_img, f_seg ──→ 2×CDBResBlock ──→ 1×1 Conv → Split(gate, value)
                                         │
                    Cross-gating: out_v = σ(gate_s) ⊙ value_v
                                  out_s = σ(gate_v) ⊙ value_s
                                         │
                              GN → SiLU → 1×1 Conv (零初始化)
                                         │
                              f_img + out_img, f_seg + out_seg
```

---

## 8. 损失函数说明

### 8.1 总损失

```
ℓ_tot = ℓ_img + ℓ_seg
```

### 8.2 图像超分辨率损失 ℓ_img

```
ℓ_img = λ₁·MSE(x̂_H, x_H) + λ₂·LPIPS(x̂_H, x_H) + ℓ_MFL(x̂_H, x_H, ŝ, s)
```

- λ₁ = 5.0, λ₂ = 10.0
- **LPIPS**: 使用 `lpips` 库的 VGG 网络，输入归一化到 [-1, 1]
- **MFL (Modified Focal Loss)**:
  ```
  ℓ_MFL = exp(∇s) ⊙ (1 - p)^γ ⊙ ||x̂_H - x_H||₁
  其中 p = ŝ⊙s + (1-ŝ)⊙(1-s), ∇ = Sobel, γ = 2.0
  ```

### 8.3 文本分割损失 ℓ_seg

```
ℓ_seg = λ₃·MSE(ŝ, s) + λ₄·Focal(ŝ, s) + Dice(ŝ, s)
```

- λ₃ = 10.0, λ₄ = 1.0
- **Focal Loss**: 标准二值 Focal Loss，γ = 2.0
- **Dice Loss**: 标准 Dice Loss，smooth = 1.0

---

## 9. 常见问题

### Q1: Kolors 下载失败？

设置 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或使用国内镜像下载：

```bash
pip install -U huggingface_hub
huggingface-cli download --resume-download Kwai-Kolors/Kolors-diffusers --local-dir ./pretrained/Kolors-diffusers
```

### Q2: 显存不足？

- 减小 `--jsd_dim`（默认 128）
- 使用混合精度训练（需自行添加 `torch.cuda.amp`）
- 使用 `gradient_checkpointing`

### Q3: 如何只用 FTSR 数据训练（不用 Real-CE）？

```bash
python train.py --use_kolors --ftsr_dir dataset/FTSR --device cuda
# Real-CE 路径不存在时自动跳过
```

### Q4: 如何验证数据构造结果？

```bash
# 构造完成后会自动生成验证网格图
# 或手动运行：
python -c "
from data.dataset import TADiSRDataset
ds = TADiSRDataset('dataset/FTSR')
print(f'样本数: {len(ds)}')
sample = ds[0]
for k, v in sample.items():
    print(f'  {k}: {v.shape if hasattr(v, \"shape\") else v}')
"
```

### Q5: CPU 调试模式和 GPU 训练模式的区别？

| 特性 | CPU 调试模式 | GPU 完整模式 |
|------|-------------|-------------|
| VAE | 4 层轻量 Conv | Kolors AutoencoderKL |
| UNet | 2 层 lightweight TACA UNet | Kolors UNet2DConditionModel + LoRA |
| 文本编码 | 随机固定张量 | ChatGLM 编码固定提示词 |
| JSD 通道 | [32, 64, 128, 128] | [128, 256, 512, 512] |
| 参数量 | ~2M | ~18GB |

### Q6: 如何适配昇腾 `torch_npu` 训练？

先安装昇腾 PyTorch 运行时（`torch_npu`），然后启动时将设备设置为 `npu`：

```bash
# 单卡
python train.py --use_kolors --ftsr_dir dataset/FTSR --device npu

# 多卡（示例：8 卡）
torchrun --nproc_per_node=8 train.py --use_kolors --ftsr_dir dataset/FTSR --device npu
```

当前 `train.py` 会自动：
- 检测 `torch_npu` 是否可用；
- 在分布式场景使用 `HCCL` 后端；
- 若未检测到 NPU 运行时，则自动回退到 CPU 并打印提示。

### Q7: 如何恢复训练？

当前版本需手动加载 checkpoint：

```python
ckpt = torch.load('checkpoints/tadisr_step5000.pt')
model.load_state_dict(ckpt['model_state_dict'], strict=False)
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_step = ckpt['step']
```

---

## 致谢

- [Kolors](https://github.com/Kwai-Kolors/Kolors) — 扩散模型骨干
- [Hi-SAM](https://github.com/ymy-k/Hi-SAM) — 文本分割
- [HAT](https://github.com/XPixelGroup/HAT) — 文本补丁超分辨率
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — 降质管线
- [LSDIR](https://huggingface.co/datasets/ofsoundof/LSDIR) — 高质量背景图像
- [CTR](https://github.com/fudan-ocr) — 中文文本识别数据集

## License

本项目仅用于学术研究。请遵守各依赖模型和数据集的许可证。
