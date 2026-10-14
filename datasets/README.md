# 数据准备 (Data Preparation)

数据集文件需自行下载，本仓库不包含数据。请按以下目录结构组织：

```
data/
├── Synapse/
│   ├── train_npz/          # 训练切片：caseXXXX_sliceXXX.npz（含 image/label 字段）
│   └── test_vol_h5/        # 测试体数据：caseXXXX.npy.h5（含 image/label 字段）
├── ACDC/
│   ├── train/              # 每个病例一个子目录：caseXXX/caseXXX.nii.gz + caseXXX_gt.nii.gz + list.txt
│   ├── valid/
│   └── test/
└── Kvasir-SEG/
    ├── TrainDataset/
    │   ├── image/          # 训练图像 (.png/.jpg)
    │   └── masks/          # 训练掩膜 (.png)
    └── TestDataset/
        └── Kvasir/
            ├── images/     # 测试图像
            └── masks/      # 测试掩膜
```

## 数据集获取

- **Synapse**：在官方 [Synapse 多器官数据集](https://www.synapse.org/#!Synapse:syn3193805/wiki/) 注册下载，并将 3D 体数据预处理为 npz（训练切片）和 h5（测试体数据）格式。
- **ACDC**：在 ACDC Challenge 官方网站注册下载（NIfTI 格式）。
- **Kvasir-SEG**：在 Kvasir-SEG 官方数据集网站下载。

## 划分列表

`lists/` 目录提供训练/测试划分列表：

- `lists_Synapse/`：Synapse 的 `train.txt` / `test_vol.txt`（12 个测试病例）
- `lists_ACDC/`：ACDC 的 `train.txt` / `test_vol.txt`
