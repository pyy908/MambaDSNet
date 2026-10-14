# MambaDSNet: An Ultra-Lightweight Dual-Stream Mamba–CNN with Cross-Stream Interaction and Edge-Aware Auxiliary Supervision for Resource-Constrained Medical Image Segmentation

**MambaDSNet** is an ultra-lightweight dual-stream network for medical image segmentation. It deeply fuses a parallel lightweight CNN branch with a Vision Mamba branch, achieving competitive segmentation performance with only **2.34M parameters** and **11.85 GFLOPs**—e.g., **80.72% Dice on Synapse** and **91.10% Dice on ACDC**.

> Official implementation of the paper submitted to *Biomedical Signal Processing and Control*.

---

## Key Ideas

### 1. Dual-Stream Architecture
- **CNN Branch**: a ConvNeXt-style lightweight convolutional block (`LightCNNBlock`) with depthwise convolutions, efficiently extracting local texture features.
- **Mamba Branch**: built on Vision Mamba (`VSSBlock` / `SS2D`), modeling global long-range dependencies to capture the overall structure of organs and lesions.

### 2. CSMI: Cross-Scanning Interaction Module
CSMI is the core fusion strategy of MambaDSNet:
- **Cross-Scan Encoding**: the spatial features of the CNN branch are encoded with direction-aware (height/width) pooling to produce bidirectional spatial attention maps.
- **Directional Gating**: a directional gating signal from the Mamba branch dynamically modulates the response strength of the Mamba features.
- **Residual Connection**: the original CNN features are preserved and added back, preventing loss of local details.

### 3. EAAT: Edge-Aware Auxiliary Task
- A parallel edge prediction head outputs an edge probability map.
- Edge ground truth is generated on the fly from the ground-truth mask via morphological gradient.
- The edge branch is supervised by a Dice-based loss (weighted by `lambda_edge`), sharpening segmentation boundaries.

---

## Architecture

```
Input Image [B, 1, H, W]
     │
     ▼
   Stem (Conv+BN+ReLU) ×2
     │
     ├─ Encoder ───────────────────────────────────────────┐
     │                                                    │
     │  Layer 1: DualStreamMambaBlock (base_c)        ──► Skip 1
     │  Down1  → [B, base_c×2, H/2, W/2]
     │  Layer 2: DualStreamMambaBlock (base_c×2)      ──► Skip 2
     │  Down2  → [B, base_c×4, H/4, W/4]
     │  Layer 3: DualStreamMambaBlock (base_c×4)      ──► Skip 3
     │  Down3  → [B, base_c×8, H/8, W/8]
     │                                                    │
     │  Bottleneck: VSSBlock (base_c×8)                   │
     │                                                    │
     └────────────────────────────────────────────────────┘
     │
     ▼
   Decoder (Upsample + Skip Concat + DoubleConv)
     │
     ├──► Segmentation Head (1×1 Conv → num_classes)  → segmentation map
     └──► Edge Head (1×1 Conv → 1)                    → edge probability map
```

**Inside `DualStreamMambaBlock`:**

```
Input ──┬──► LightCNNBlock (CNN branch) ──┐
        │                                 ├──► CSMI Fusion ──► Output
        └──► VSSBlock (Mamba branch) ─────┘
```

---

## Supported Datasets

| Dataset | Modality | Classes | Task |
|---|---|---|---|
| **Synapse** | Abdominal CT | 9 (8 organs + background) | Multi-organ segmentation |
| **ACDC** | Cardiac MRI | 4 (LV, RV, MYO + background) | Cardiac segmentation |
| **Kvasir-SEG** | Endoscopic images | 1 (polyp) | Polyp segmentation |

---

## Data Preparation

Please download each dataset from its official repository:

| Dataset | Official Source |
|---|---|
| Synapse (abdominal multi-organ CT) | https://www.synapse.org/Synapse:syn3193805 (registration required) |
| ACDC (cardiac MRI) | https://www.creatis.insa-lyon.fr/Challenge/acdc/ |
| Kvasir-SEG (endoscopic polyps) | https://datasets.simula.no/kvasir-seg/ |
| M&Ms (cardiac MRI; used for zero-shot evaluation) | https://www.ub.edu/mnms/ |

Place the downloaded data following the layout expected by `datasets/` (see `datasets/README.md` for details). Training/validation splits are provided under `lists/`.

---

## Supported Models

### Main Model
| Name | Description |
|---|---|
| `mambadsnet` | Full dual-stream model with CSMI fusion and edge-aware supervision |

### Ablation Variants
| Name | Description |
|---|---|
| `mamba_ab1` | Mamba-only (no CNN branch) |
| `mamba_ab2` | Mamba + CNN without CSMI (simple summation) |
| `mamba_csmi_nodir` | CSMI without directional gating |
| `mamba_csmi_nores` | CSMI without the residual connection |
| `mamba_ca_self` | Independent Coordinate Attention (CA) self-attention with summation fusion |

### Baselines
The baselines compared in the paper (U-Net, Attention U-Net, TransUNet, Swin-Unet, VM-UNet, U-Mamba, etc.) follow their official implementations and are **not** included in this repository. Training/evaluation protocols and hyperparameter settings are detailed in the paper.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### MambaDSNet

```bash
# Synapse multi-organ segmentation
python train.py --model mambadsnet --dataset Synapse --max_epochs 150 --batch_size 24 --base_lr 0.02

# ACDC cardiac segmentation
python train.py --model mambadsnet --dataset ACDC --max_epochs 200 --num_classes 4 --batch_size 8 --base_lr 0.01

# Kvasir polyp segmentation
python train.py --model mambadsnet --dataset Kvasir --max_epochs 150 --num_classes 1 --batch_size 24 --base_lr 1e-4
```

### Ablation Studies

```bash
# Mamba-only
python train.py --model mamba_ab1 --dataset Synapse --max_epochs 150 --batch_size 24

# Mamba + CNN (simple summation)
python train.py --model mamba_ab2 --dataset Synapse --max_epochs 150 --batch_size 24

# CSMI without directional gating
python train.py --model mamba_csmi_nodir --dataset Synapse --max_epochs 150 --batch_size 24

# CSMI without residual connection
python train.py --model mamba_csmi_nores --dataset Synapse --max_epochs 150 --batch_size 24

# Independent CA self-attention
python train.py --model mamba_ca_self --dataset Synapse --max_epochs 150 --batch_size 24
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--model` | Model architecture | `mambadsnet` |
| `--dataset` | Dataset name | `Synapse` |
| `--max_epochs` | Maximum training epochs | `150` |
| `--batch_size` | Batch size per GPU | `24` |
| `--base_lr` | Initial learning rate | `0.02` (`1e-4` for Kvasir) |
| `--base_c` | Base channel width | `32` |
| `--lambda_edge` | Edge loss weight (0 = disabled) | `0.2` |
| `--img_size` | Input image size | `224` |
| `--seed` | Random seed | `1234` |
| `--disable_amp` | Disable mixed-precision training | `False` |
| `--num_workers` | Number of data-loading workers | `8` |

---

## Evaluation

```bash
# Synapse
python test.py --arch mambadsnet --dataset Synapse

# ACDC (3D volumetric inference)
python test.py --arch mambadsnet --dataset ACDC

# Kvasir (standalone script)
python test_kvasir.py --model_path <path_to_checkpoint>
```

### Metrics
- **Dice Score (DSC)**
- **95% Hausdorff Distance (HD95)**
- **Average Symmetric Surface Distance (ASSD)**
- **Recall / Sensitivity**
- **Statistical significance** — `ablation_significance_test.py`

> **Significance testing.** `ablation_significance_test.py` first requires the test logs of each variant generated by `test.py` under `test_log/test/`, from which it parses the per-case metrics.

> **Evaluation protocol.** Metric computation follows the evaluation convention used by the TransUNet / Swin-Unet codebase for organs absent from a case's ground truth (empty GT): when the ground truth is empty **but the prediction is non-empty**, the sample is scored as Dice = 1 and HD95/ASSD = 0; when both the ground truth and the prediction are empty, the sample is scored as 0. This keeps all reported numbers directly comparable with the baselines (TransUNet, Swin-Unet, etc.).

---

## Project Structure

```
MambaDSNet/
├── mamba_dsnet.py                 # Main model (Dual-Stream + CSMI + Edge head)
├── mamba.py                       # Ablation variants: Mamba-Only / Mamba+CNN
├── vss_block.py                   # VSSBlock + SS2D (Vision Mamba)
├── csmi_ablation.py               # CSMI ablations: NoDir / NoResidual / CASelf
│
├── train.py                       # Unified training script
├── trainer_kvasir.py              # Kvasir trainer
├── train_unet_kvasir.py           # U-Net Kvasir trainer (baseline)
├── test.py                        # Unified testing script (Synapse / ACDC)
├── test_kvasir.py                 # Kvasir testing
├── test_unet_kvasir.py            # U-Net Kvasir testing (baseline)
├── ablation_significance_test.py  # Ablation significance testing
├── utils.py                       # Utility functions (DiceLoss, etc.)
│
├── datasets/                      # Dataset loaders
├── lists/                         # Dataset split lists
│
├── requirements.txt               # Dependencies
├── LICENSE                        # License
└── README.md
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{mambadsnet,
  title   = {MambaDSNet: An Ultra-Lightweight Dual-Stream Mamba-CNN with Cross-Stream Interaction and Edge-Aware Auxiliary Supervision for Resource-Constrained Medical Image Segmentation},
  author  = {Chu, Shu-Chuan and Pan, Yiyi and Zhao, Ming and Wang, Jiahua and Pan, Jeng-Shyang},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## Acknowledgements

This project was developed with reference to the following open-source works:

- [Mamba](https://github.com/state-spaces/mamba) — state space models
- [VMamba](https://github.com/MzeroMiko/VMamba) — Vision Mamba
- [VM-UNet](https://github.com/nanase1025/VM-UNet) — VMamba U-Net
- [U-Mamba](https://github.com/bowang-lab/U-Mamba) — Mamba for medical segmentation
- [TransUNet](https://github.com/Beckschen/TransUNet) — Transformer U-Net
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) — Swin U-Net
- [MONAI](https://monai.io/) — medical imaging deep learning framework

---

## License

This project is released under the license provided in the [LICENSE](LICENSE) file.
