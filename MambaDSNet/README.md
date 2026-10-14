# MambaDSNet: Dual-Stream Mamba-CNN Network with Cross-Scan Modulated Interaction for Medical Image Segmentation

MambaDSNet 是一种用于医学图像分割的新型双流网络，通过并行 CNN 分支与 Vision Mamba 分支的深度融合，在 Synapse、ACDC 和 Kvasir-SEG 等多个医学图像分割基准上取得了优异的性能。

---

## 核心思想

### 1. 双流并行架构 (Dual-Stream Architecture)

- **CNN 分支**: 采用 ConvNeXt 风格的轻量级 CNN（`LightCNNBlock`），使用深度可分离卷积高效提取局部纹理特征
- **Mamba 分支**: 基于 Vision Mamba (`VSSBlock` / `SS2D`) 建模全局长程依赖，捕捉器官/病灶的整体结构信息

### 2. CSMI 融合模块 (Cross-Scan Modulated Interaction)

CSMI 是本模型的核心创新融合策略：

- **Cross-Scan Encoding**: 将 CNN 分支的空间特征沿高度和宽度方向进行交叉扫描编码，生成双向空间注意力图
- **Directional Gating**: 利用 Mamba 分支的方向门控信号，动态调制 Mamba 特征的响应强度
- **Residual Connection**: 保留 CNN 分支的原始特征，确保局部细节不丢失

### 3. 边缘感知损失 (Edge-Aware Loss)

- 并行的边缘预测头输出边缘概率图
- 使用形态学梯度从真实标签自动生成边缘真值
- 通过 Binary Dice Loss 监督边缘预测，提升分割边界的精确性

---

## 网络架构

```
Input Image [B, 1, H, W]
     │
     ▼
   Stem (Conv+BN+ReLU) ×2
     │
     ├─ Encoder ───────────────────────────────────────────┐
     │                                                    │
     │  Layer 1: DualStreamMambaBlock (base_c)        ──► Skip 1
     │  Down1  → [B, base_c×2, H/2, W/2]              ──► Skip 2
     │  Layer 2: DualStreamMambaBlock (base_c×2)      ──► Skip 3
     │  Down2  → [B, base_c×4, H/4, W/4]
     │  Layer 3: DualStreamMambaBlock (base_c×4)
     │  Down3  → [B, base_c×8, H/8, W/8]
     │                                                    │
     │  Bottleneck: VSSBlock (base_c×8)                   │
     │                                                    │
     └────────────────────────────────────────────────────┘
     │
     ▼
   Decoder (Upsample + Skip Concat + DoubleConv)
     │
     ├──► Segmentation Head (1×1 Conv → num_classes)  → 分割结果
     └──► Edge Head (1×1 Conv → 1)                    → 边缘概率图
```

**DualStreamMambaBlock 内部结构：**

```
Input ──┬──► LightCNNBlock (CNN 分支) ──┐
        │                               ├──► CSMI Fusion ──► Output
        └──► VSSBlock (Mamba 分支) ─────┘
```

---

## 支持的数据集

| 数据集 | 模态 | 类别数 | 任务 |
|--------|------|--------|------|
| **Synapse** | 腹部 CT | 9 (8 器官 + 背景) | 多器官分割 |
| **ACDC** | 心脏 MRI | 4 (LV, RV, MYO + 背景) | 心脏分割 |
| **Kvasir-SEG** | 内窥镜图像 | 1 (息肉) | 息肉分割 |

---

## 支持的模型

### 主模型
| 参数 | 说明 |
|------|------|
| `mambadsnet` | 完整的双流 CSMI 融合模型（含边缘感知） |

### 消融实验变体
| 参数 | 说明 |
|------|------|
| `mamba_ab1` | 纯 Mamba（无 CNN 分支） |
| `mamba_ab2` | Mamba + CNN（无 CSMI 融合，简单求和） |
| `mamba_csmi_nodir` | CSMI 无方向门控 |
| `mamba_csmi_nores` | CSMI 无残差连接 |
| `mamba_ca_self` | 标准交叉注意力 + 求和融合 |

### 对比方法（Baselines）

对比实验所使用的 baseline（U-Net、Attention U-Net、TransUNet、SwinUNet、VMUNet、U-Mamba 等）均使用各自官方仓库实现，本仓库仅包含 MambaDSNet 及其消融变体，不包含 baseline 的实现代码。训练/评估协议与超参配置详见论文正文。

---

## 环境配置

### 依赖

```bash
pip install -r requirements.txt
```

---

## 训练

### MambaDSNet

```bash
# Synapse 多器官分割
python train.py --model mambadsnet --dataset Synapse --max_epochs 150 --batch_size 24 --base_lr 0.02

# ACDC 心脏分割
python train.py --model mambadsnet --dataset ACDC --max_epochs 200 --num_classes 4 --batch_size 8 --base_lr 0.01

# Kvasir 息肉分割
python train.py --model mambadsnet --dataset Kvasir --max_epochs 150 --num_classes 1 --batch_size 24 --base_lr 1e-4
```

### 消融实验

```bash
# 纯 Mamba
python train.py --model mamba_ab1 --dataset Synapse --max_epochs 150 --batch_size 24

# Mamba + CNN
python train.py --model mamba_ab2 --dataset Synapse --max_epochs 150 --batch_size 24

# CSMI 无方向门控
python train.py --model mamba_csmi_nodir --dataset Synapse --max_epochs 150 --batch_size 24

# CSMI 无残差连接
python train.py --model mamba_csmi_nores --dataset Synapse --max_epochs 150 --batch_size 24

# 标准交叉注意力
python train.py --model mamba_ca_self --dataset Synapse --max_epochs 150 --batch_size 24
```

### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型架构 | `mambadsnet` |
| `--dataset` | 数据集 | `Synapse` |
| `--max_epochs` | 最大训练轮数 | `150` |
| `--batch_size` | 每 GPU batch 大小 | `24` |
| `--base_lr` | 初始学习率 | `0.02`（Kvasir 为 `1e-4`）|
| `--base_c` | 基础通道数 | `32` |
| `--lambda_edge` | 边缘损失权重 (0 = 关闭) | `0.2` |
| `--img_size` | 输入图像大小 | `224` |
| `--seed` | 随机种子 | `1234` |
| `--disable_amp` | 禁用混合精度训练 | `False` |
| `--num_workers` | 数据加载线程数 | `8` |

---

## 测试

```bash
# Synapse 测试
python test.py --arch mambadsnet --dataset Synapse

# ACDC 测试 (3D 体积推理)
python test.py --arch mambadsnet --dataset ACDC

# Kvasir 息肉测试（独立脚本）
python test_kvasir.py --model_path <模型权重路径>
```

### 评估指标

- **Dice Score (DSC)**: 分割重叠度
- **Hausdorff Distance 95% (HD95)**: 边界距离
- **Average Symmetric Surface Distance (ASSD)**: 平均表面距离
- **Recall / Sensitivity**: 召回率
- **统计显著性检验**: `ablation_significance_test.py`（消融实验显著性）

> **显著性检验说明**：`ablation_significance_test.py` 需先运行 `test.py` 生成 `test_log/test/` 下的测试日志后，才能解析各变体的 per-case 指标。

> **评估口径说明**：指标计算遵循 TransUNet 的评估口径——对某个器官在单个病例中缺失（GT 为空）但被预测为前景的情况，Dice 记 1、HD95/ASSD 记 0，以保证与对比方法（TransUNet、SwinUNet 等）口径一致。

---

## 项目结构

```
MambaDSNet/
├── mamba_dsnet.py           # MambaDSNet 主模型 (DualStream + CSMI + Edge)
├── mamba.py                 # 消融变体: Mamba-Only / Mamba-CNN
├── vss_block.py             # VSSBlock + SS2D (Vision Mamba)
├── csmi_ablation.py         # CSMI 消融: NoDir / NoResidual / CASelf
│
├── train.py                 # 统一训练脚本
├── trainer_kvasir.py        # Kvasir 训练器
├── train_unet_kvasir.py     # U-Net Kvasir 训练器（baseline）
│
├── test.py                  # 统一测试脚本 (Synapse/ACDC)
├── test_kvasir.py           # Kvasir 测试
├── test_unet_kvasir.py      # U-Net Kvasir 测试（baseline）
│
├── ablation_significance_test.py  # 消融显著性检验
├── utils.py                 # 工具函数 (DiceLoss 等)
│
├── datasets/                # 数据集加载
├── lists/                   # 数据集划分列表
│
├── requirements.txt         # 依赖项
├── LICENSE                  # 许可证
└── README.md
```

---

## 引用

```bibtex
@article{mambadsnet,
  title={MambaDSNet: Dual-Stream Mamba-CNN Network with Cross-Scan Modulated Interaction for Medical Image Segmentation},
  author={},
  journal={},
  year={2025}
}
```

---

## 致谢

本项目参考了以下开源工作：

- [Mamba](https://github.com/state-spaces/mamba) — 状态空间模型
- [VMamba](https://github.com/MzeroMiko/VMamba) — Vision Mamba
- [VMUNet](https://github.com/nanase1025/VM-UNet) — VMamba U-Net
- [U-Mamba](https://github.com/bowang-lab/U-Mamba) — 医学分割 Mamba
- [TransUNet](https://github.com/Beckschen/TransUNet) — Transformer U-Net
- [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet) — Swin U-Net
- [MONAI](https://monai.io/) — 医学影像深度学习框架
