"""
CSMI 消融实验模块，提供三个消融变体，分别量化 CSMI 的三个设计维度：

1. CSMI without Direction Gate (mamba_csmi_nodir)
   - 去掉方向/边界门控 (dir_att)，保留跨流 CA + 残差 CNN

2. CSMI without Residual CNN (mamba_csmi_nores)
   - 去掉残差 CNN 基底 (+ f_c)，保留跨流 CA + 方向门控

3. CA Self-Attention + Sum Fusion (mamba_ca_self)
   - 各分支独立 CA 自注意力 + 简单求和融合

参考: Coordinate Attention (Hou et al., CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vss_block import VSSBlock
from mamba import LightCNNBlock, BaseAblationUNet


# 标准 Coordinate Attention 模块 (Hou et al., CVPR 2021)
class CoordAttn(nn.Module):
    """
    标准 Coordinate Attention，来自论文:
    "Coordinate Attention for Efficient Mobile Network Design" (Hou et al., CVPR 2021)

    对自身输入做两个方向 (H, W) 的全局池化 → 变换 → 拆分 → Sigmoid 门控。
    这是纯 self-attention 版本，用作对照基线。
    """

    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 两个方向的全局编码
        x_h = F.adaptive_avg_pool2d(x, (h, 1))       # [B, C, H, 1]
        x_w = F.adaptive_avg_pool2d(x, (1, w))       # [B, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)                 # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)              # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))         # [B, mip, H+W, 1]

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                 # [B, mip, 1, W]

        att_h = torch.sigmoid(self.conv_h(x_h))       # [B, C, H, 1]
        att_w = torch.sigmoid(self.conv_w(x_w))       # [B, C, 1, W]

        return identity * att_h * att_w


# 消融 A：CSMI 去掉方向门控
class CSMI_Fusion_NoDir(nn.Module):
    """
    CSMI 去掉 Direction Gate 的版本。

    与完整 CSMI 的唯一区别：不接收/不使用 dir_att。
    核心仍是: f_c 做 Coordinate Attention → 门控 f_m，残差加回 f_c。

    这个变体本质上就是 "Cross-Stream Coordinate Attention + Residual"。
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        self.conv_cnn = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv_mamba = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.scan_transform = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish()
        )

        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, bias=False)

        # 无 dir_gate（与完整 CSMI 的区别）

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_cnn, x_mamba):
        """不再接收 dir_att 参数"""
        f_c = self.conv_cnn(x_cnn)
        f_m = self.conv_mamba(x_mamba)
        _, _, H, W = f_c.shape

        # Cross-Scan Encoding（与完整 CSMI 完全一致）
        x_h = F.adaptive_avg_pool2d(f_c, (H, 1))
        x_w = F.adaptive_avg_pool2d(f_c, (1, W))

        x_w_perm = x_w.permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w_perm], dim=2)

        x_cat = self.scan_transform(x_cat)
        x_h_new, x_w_new = torch.split(x_cat, [H, W], dim=2)
        x_w_new = x_w_new.permute(0, 1, 3, 2)

        att_h = torch.sigmoid(self.conv_h(x_h_new))
        att_w = torch.sigmoid(self.conv_w(x_w_new))

        # 差异：无 dir_att 门控调制
        out = f_m * att_h * att_w + f_c
        return self.final_conv(out)


class DualStreamMambaBlock_NoDir(nn.Module):
    """
    使用 CSMI_Fusion_NoDir 的双流块。
    与 DualStreamMambaBlock 结构一致，仅融合模块不同。
    """

    def __init__(self, dim, depth=1):
        super().__init__()
        self.mamba_branch = nn.ModuleList([
            VSSBlock(hidden_dim=dim, dropout=0.1)
            for _ in range(depth)
        ])
        self.cnn_branch = nn.ModuleList([
            LightCNNBlock(dim, dropout=0.1)
            for _ in range(depth)
        ])
        self.fusion = CSMI_Fusion_NoDir(dim)

    def forward(self, x):
        x_m = x
        for blk in self.mamba_branch:
            x_m = blk(x_m)

        x_c = x
        for blk in self.cnn_branch:
            x_c = blk(x_c)

        # 不传 dir_att
        x_out = self.fusion(x_c, x_m)
        return x_out


# 消融 B：CSMI 去掉残差 CNN 基底
class CSMI_Fusion_NoResidual(nn.Module):
    """
    CSMI 去掉残差 CNN 基底 (+ f_c) 的版本。

    与完整 CSMI 的区别:
    - 保留方向门控 (dir_att)
    - 去掉残差连接: out = f_m * att_h * att_w （无 + f_c）
    - 量化 "残差 CNN 基底" 对融合的贡献
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        self.conv_cnn = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv_mamba = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.scan_transform = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish()
        )

        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, bias=False)

        self.dir_gate = nn.Conv2d(in_channels, in_channels, 1, bias=True)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_cnn, x_mamba, dir_att=None):
        f_c = self.conv_cnn(x_cnn)
        f_m = self.conv_mamba(x_mamba)
        _, _, H, W = f_c.shape

        # Cross-Scan Encoding
        x_h = F.adaptive_avg_pool2d(f_c, (H, 1))
        x_w = F.adaptive_avg_pool2d(f_c, (1, W))

        x_w_perm = x_w.permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w_perm], dim=2)

        x_cat = self.scan_transform(x_cat)
        x_h_new, x_w_new = torch.split(x_cat, [H, W], dim=2)
        x_w_new = x_w_new.permute(0, 1, 3, 2)

        att_h = torch.sigmoid(self.conv_h(x_h_new))
        att_w = torch.sigmoid(self.conv_w(x_w_new))

        if dir_att is not None:
            gate = torch.sigmoid(self.dir_gate(dir_att))
            f_m = f_m * (1.0 + gate)

        # 差异：无 + f_c 残差连接
        out = f_m * att_h * att_w
        return self.final_conv(out)


class DualStreamMambaBlock_NoResidual(nn.Module):
    """
    使用 CSMI_Fusion_NoResidual 的双流块。
    去掉残差 CNN 基底，仅保留被 CA 门控的 Mamba 特征。
    """

    def __init__(self, dim, depth=1):
        super().__init__()
        self.mamba_branch = nn.ModuleList([
            VSSBlock(hidden_dim=dim, dropout=0.1)
            for _ in range(depth)
        ])
        self.cnn_branch = nn.ModuleList([
            LightCNNBlock(dim, dropout=0.1)
            for _ in range(depth)
        ])
        self.fusion = CSMI_Fusion_NoResidual(dim)

    def forward(self, x):
        x_m = x
        dir_att = None
        for blk in self.mamba_branch:
            x_m = blk(x_m)
            dir_att = getattr(blk, "last_dir_att", None)

        x_c = x
        for blk in self.cnn_branch:
            x_c = blk(x_c)

        x_out = self.fusion(x_c, x_m, dir_att)
        return x_out


# 消融 C：标准 CA 自注意力 + 简单求和融合
class CA_SelfAttn_Fusion(nn.Module):
    """
    对照模块：各分支独立做标准 Coordinate Attention（self-attention），
    然后简单求和 + 1x1 Conv 融合。

    与 CSMI 的核心区别:
    - CSMI: 用 CNN 特征生成注意力 → 门控 Mamba 特征（跨流 cross-attention）
    - 本模块: CNN 自己做 CA, Mamba 自己做 CA → 求和融合（各自 self-attention）
    """

    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.ca_cnn = CoordAttn(in_channels, reduction=reduction)
        self.ca_mamba = CoordAttn(in_channels, reduction=reduction)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_cnn, x_mamba):
        x_c = self.ca_cnn(x_cnn)
        x_m = self.ca_mamba(x_mamba)
        return self.fuse(x_c + x_m)


class DualStreamMambaBlock_CA_Self(nn.Module):
    """
    使用 CA_SelfAttn_Fusion 的双流块。
    各分支独立做 CA 自注意力，然后简单求和融合。
    """

    def __init__(self, dim, depth=1):
        super().__init__()
        self.mamba_branch = nn.ModuleList([
            VSSBlock(hidden_dim=dim, dropout=0.1)
            for _ in range(depth)
        ])
        self.cnn_branch = nn.ModuleList([
            LightCNNBlock(dim, dropout=0.1)
            for _ in range(depth)
        ])
        self.fusion = CA_SelfAttn_Fusion(dim)

    def forward(self, x):
        x_m = x
        for blk in self.mamba_branch:
            x_m = blk(x_m)

        x_c = x
        for blk in self.cnn_branch:
            x_c = blk(x_c)

        x_out = self.fusion(x_c, x_m)
        return x_out


# 完整消融模型（基于 BaseAblationUNet 架构）
class MambaAblationCSMINoDir(BaseAblationUNet):
    """
    消融 A (CSMI without Direction Gate):
    Mamba + CNN + CSMI_NoDir (无方向门控的跨流 Coordinate Attention + 残差)

    公式: out = f_m * att_h * att_w + f_c  (无 dir_att 调制)
    """

    def __init__(self, in_ch=1, num_classes=4, base_c=32):
        super().__init__(
            in_ch=in_ch,
            num_classes=num_classes,
            base_c=base_c,
            block_factory=lambda dim: DualStreamMambaBlock_NoDir(dim, depth=1),
        )


class MambaAblationCSMINoResidual(BaseAblationUNet):
    """
    消融 B (CSMI without Residual CNN):
    Mamba + CNN + CSMI_NoResidual (无残差 CNN 基底的跨流 Coordinate Attention + 方向门控)

    公式: out = f_m * att_h * att_w  (无 + f_c)
    """

    def __init__(self, in_ch=1, num_classes=4, base_c=32):
        super().__init__(
            in_ch=in_ch,
            num_classes=num_classes,
            base_c=base_c,
            block_factory=lambda dim: DualStreamMambaBlock_NoResidual(dim, depth=1),
        )


class MambaAblationCASelf(BaseAblationUNet):
    """
    消融 C (Standard CA Self-Attention + Sum Fusion):
    Mamba + CNN + 各分支独立 CA + 简单求和融合

    公式: out = CA_self(CNN) + CA_self(Mamba)，然后 1×1 Conv 融合
    """

    def __init__(self, in_ch=1, num_classes=4, base_c=32):
        super().__init__(
            in_ch=in_ch,
            num_classes=num_classes,
            base_c=base_c,
            block_factory=lambda dim: DualStreamMambaBlock_CA_Self(dim, depth=1),
        )


# 测试代码
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 测试标准 Coordinate Attention
    ca = CoordAttn(64).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = ca(x)
    assert out.shape == x.shape, f"CoordAttn shape mismatch: {out.shape} vs {x.shape}"
    print("CoordAttn: 测试通过")

    # 测试 CSMI_Fusion_NoDir
    fusion_nodir = CSMI_Fusion_NoDir(64).to(device)
    x_c = torch.randn(2, 64, 32, 32).to(device)
    x_m = torch.randn(2, 64, 32, 32).to(device)
    out_nodir = fusion_nodir(x_c, x_m)
    assert out_nodir.shape == x_c.shape, f"CSMI_Fusion_NoDir shape mismatch"
    print("CSMI_Fusion_NoDir: 测试通过")

    # 测试 CSMI_Fusion_NoResidual
    fusion_nores = CSMI_Fusion_NoResidual(64).to(device)
    dir_dummy = torch.randn(2, 64, 32, 32).to(device)
    out_nores = fusion_nores(x_c, x_m, dir_dummy)
    assert out_nores.shape == x_c.shape, f"CSMI_Fusion_NoResidual shape mismatch"
    print("CSMI_Fusion_NoResidual: 测试通过")

    # 测试 CA_SelfAttn_Fusion
    fusion_ca = CA_SelfAttn_Fusion(64).to(device)
    out_ca = fusion_ca(x_c, x_m)
    assert out_ca.shape == x_c.shape, f"CA_SelfAttn_Fusion shape mismatch"
    print("CA_SelfAttn_Fusion: 测试通过")

    # 测试 DualStreamMambaBlock_NoDir
    blk_nodir = DualStreamMambaBlock_NoDir(64).to(device)
    out_blk_nodir = blk_nodir(x)
    assert out_blk_nodir.shape == x.shape, f"DualStreamMambaBlock_NoDir shape mismatch"
    print("DualStreamMambaBlock_NoDir: 测试通过")

    # 测试 DualStreamMambaBlock_NoResidual
    blk_nores = DualStreamMambaBlock_NoResidual(64).to(device)
    out_blk_nores = blk_nores(x)
    assert out_blk_nores.shape == x.shape, f"DualStreamMambaBlock_NoResidual shape mismatch"
    print("DualStreamMambaBlock_NoResidual: 测试通过")

    # 测试 DualStreamMambaBlock_CA_Self
    blk_ca = DualStreamMambaBlock_CA_Self(64).to(device)
    out_blk_ca = blk_ca(x)
    assert out_blk_ca.shape == x.shape, f"DualStreamMambaBlock_CA_Self shape mismatch"
    print("DualStreamMambaBlock_CA_Self: 测试通过")

    # 测试完整模型
    model_nodir = MambaAblationCSMINoDir(in_ch=1, num_classes=9, base_c=32).to(device)
    img = torch.randn(1, 1, 224, 224).to(device)
    out_nodir_model = model_nodir(img)
    assert out_nodir_model.shape == (1, 9, 224, 224), \
        f"MambaAblationCSMINoDir shape: {out_nodir_model.shape}, expected (1,9,224,224)"
    print("MambaAblationCSMINoDir: 测试通过")

    model_nores = MambaAblationCSMINoResidual(in_ch=1, num_classes=9, base_c=32).to(device)
    out_nores_model = model_nores(img)
    assert out_nores_model.shape == (1, 9, 224, 224), \
        f"MambaAblationCSMINoResidual shape: {out_nores_model.shape}, expected (1,9,224,224)"
    print("MambaAblationCSMINoResidual: 测试通过")

    model_ca = MambaAblationCASelf(in_ch=1, num_classes=9, base_c=32).to(device)
    out_ca_model = model_ca(img)
    assert out_ca_model.shape == (1, 9, 224, 224), \
        f"MambaAblationCASelf shape: {out_ca_model.shape}, expected (1,9,224,224)"
    print("MambaAblationCASelf: 测试通过")

    # 统计参数量对比
    from mamba_dsnet import CSMI_Fusion
    csmi_full = CSMI_Fusion(64)
    csmi_nodir = CSMI_Fusion_NoDir(64)
    csmi_nores = CSMI_Fusion_NoResidual(64)
    ca_self = CA_SelfAttn_Fusion(64)

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    print(f"\n{'='*50}")
    print(f"参数量对比 (in_channels=64):")
    print(f"  CSMI_Fusion (full):       {count_params(csmi_full):>8,}")
    print(f"  CSMI_Fusion_NoDir:        {count_params(csmi_nodir):>8,}")
    print(f"  CSMI_Fusion_NoResidual:   {count_params(csmi_nores):>8,}")
    print(f"  CA_SelfAttn_Fusion:       {count_params(ca_self):>8,}")
    print(f"  MambaAblationCSMINoDir:   {count_params(model_nodir):>8,}")
    print(f"  MambaAblationCSMINoResidual: {count_params(model_nores):>8,}")
    print(f"  MambaAblationCASelf:      {count_params(model_ca):>8,}")
    print(f"所有消融模块测试通过！")
