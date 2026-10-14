import torch
import torch.nn as nn
import torch.nn.functional as F

# VSSBlock（Vision Mamba 块，简化实现）
from vss_block import VSSBlock
# 轻量 CNN 块与解码器卷积（定义在 mamba.py，供主模型与消融变体共用）
from mamba import LightCNNBlock, DoubleConv

# CSMI 融合模块
class CSMI_Fusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        # 投影层
        self.conv_cnn = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv_mamba = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # 扫描变换
        self.scan_transform = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Hardswish()
        )

        # 注意力生成
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, bias=False)

        # 将方向/边界门控融合进来
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

        # Cross-Scan Encoding (来自 CNN 分支的空间流)
        x_h = F.adaptive_avg_pool2d(f_c, (H, 1))
        x_w = F.adaptive_avg_pool2d(f_c, (1, W))

        x_w_perm = x_w.permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w_perm], dim=2)

        x_cat = self.scan_transform(x_cat)
        x_h_new, x_w_new = torch.split(x_cat, [H, W], dim=2)
        x_w_new = x_w_new.permute(0, 1, 3, 2)

        att_h = torch.sigmoid(self.conv_h(x_h_new))
        att_w = torch.sigmoid(self.conv_w(x_w_new))

        # 如果有多方向/边界门控，则调制 Mamba 特征
        if dir_att is not None:
            gate = torch.sigmoid(self.dir_gate(dir_att))
            f_m = f_m * (1.0 + gate)

        out = f_m * att_h * att_w + f_c
        return self.final_conv(out)


# 双流 Mamba Block：并行 CNN + Mamba，经 CSMI 融合
class DualStreamMambaBlock(nn.Module):
    """并行 CNN + Mamba 双流模块，通过 CSMI 融合，可堆叠使用。"""

    def __init__(self, dim, depth=1):
        super().__init__()
        # 分支 1：Vision Mamba (VSS)
        self.mamba_branch = nn.ModuleList([
            VSSBlock(hidden_dim=dim, dropout=0.1)
            for _ in range(depth)
        ])

        # 分支 2：轻量 CNN（ConvNeXt）
        self.cnn_branch = nn.ModuleList([
            LightCNNBlock(dim, dropout=0.1)
            for _ in range(depth)
        ])

        self.fusion = CSMI_Fusion(dim)

    def forward(self, x):
        # Mamba 分支
        x_m = x
        dir_att = None
        for blk in self.mamba_branch:
            x_m = blk(x_m)
            dir_att = getattr(blk, "last_dir_att", None)  # 取最后一层方向/边界提示

        # CNN 分支
        x_c = x
        for blk in self.cnn_branch:
            x_c = blk(x_c)

        # 融合
        x_out = self.fusion(x_c, x_m, dir_att)
        return x_out


# 主模型：MambaDSNet（改进版 U-Net）
class MambaDSNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=9, base_c=32):
        super().__init__()
        self.num_classes = num_classes

        # --- Stem（初始特征提取，保持分辨率）---
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True)
        )

        # --- Encoder（以下分辨率按默认 224×224 输入）---
        # Layer 1: [224x224] - 双流
        self.encoder1 = DualStreamMambaBlock(base_c, depth=1)
        self.down1 = nn.Conv2d(base_c, base_c * 2, 3, stride=2, padding=1)  # 下采样

        # Layer 2: [112x112] - 双流
        self.encoder2 = DualStreamMambaBlock(base_c * 2, depth=1)
        self.down2 = nn.Conv2d(base_c * 2, base_c * 4, 3, stride=2, padding=1)

        # Layer 3: [56x56] - 双流
        self.encoder3 = DualStreamMambaBlock(base_c * 4, depth=1)
        self.down3 = nn.Conv2d(base_c * 4, base_c * 8, 3, stride=2, padding=1)

        # Layer 4: [28x28] - Bottleneck (通常只用 Mamba 获取最大感受野)
        self.bottleneck = VSSBlock(hidden_dim=base_c * 8, dropout=0.1)

        # --- Decoder（U-Net 风格） ---
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(base_c * 8 + base_c * 4, base_c * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(base_c * 4 + base_c * 2, base_c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(base_c * 2 + base_c, base_c)

        self.final_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)
        # Edge prediction head: parallel 1x1 conv producing edge probability map
        self.edge_conv = nn.Conv2d(base_c, 1, kernel_size=1)

    def forward(self, x):
        # 1. Stem
        x1 = self.stem(x)  # [B, 32, 224, 224]

        # 2. Encoder
        f1 = self.encoder1(x1)  # Skip 1
        d1 = self.down1(f1)  # [B, 64, 112, 112]

        f2 = self.encoder2(d1)  # Skip 2
        d2 = self.down2(f2)  # [B, 128, 56, 56]

        f3 = self.encoder3(d2)  # Skip 3
        d3 = self.down3(f3)  # [B, 256, 28, 28]

        # 3. Bottleneck
        bot = self.bottleneck(d3)

        # 4. Decoder
        u3 = self.up3(bot)
        cat3 = torch.cat([u3, f3], dim=1)  # Skip Connection
        dec3 = self.conv3(cat3)

        u2 = self.up2(dec3)
        cat2 = torch.cat([u2, f2], dim=1)
        dec2 = self.conv2(cat2)

        u1 = self.up1(dec2)
        cat1 = torch.cat([u1, f1], dim=1)
        dec1 = self.conv1(cat1)

        # Outputs: segmentation logits and edge probability map
        out = self.final_conv(dec1)
        edge = self.edge_conv(dec1)
        return out, edge  # 分割 logits + 边缘图


# 测试代码
if __name__ == "__main__":
    # 1. 定义超参数
    BATCH_SIZE = 2
    INPUT_CHANNELS = 1  # CT图通常是1通道
    NUM_CLASSES = 9  # Synapse数据集通常是8个器官+1个背景 = 9
    HEIGHT = 512  # 常见的医学图像尺寸
    WIDTH = 512

    # 2. 创建模拟输入数据 (随机噪声)
    # 形状: [Batch, Channel, Height, Width]
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)

    # 3. 实例化模型
    # 有显卡用 cuda，否则回退到 cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaDSNet(in_ch=INPUT_CHANNELS, num_classes=NUM_CLASSES).to(device)
    dummy_input = dummy_input.to(device)

    print(f"正在测试模型输入输出...")
    print(f"输入形状: {dummy_input.shape}")

    # 4. 前向传播
    try:
        seg, edge = model(dummy_input)

        # 5. 验证形状
        print(f"分割输出形状: {seg.shape}（边缘: {edge.shape}）")

        expected_shape = torch.Size([BATCH_SIZE, NUM_CLASSES, HEIGHT, WIDTH])
        if seg.shape == expected_shape:
            print("测试通过：分割输出形状符合 [B, Num_Classes, H, W]")
        else:
            print(f"测试失败：预期 {expected_shape}，实际 {seg.shape}")

    except Exception as e:
        print(f"运行报错: {e}")