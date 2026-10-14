import torch
import torch.nn as nn

from vss_block import VSSBlock


class LightCNNBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.dropout = nn.Identity() if dropout <= 0 else nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.dropout(x)


class MambaOnlyBlock(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim, dropout=0.1)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class MambaCNNNoCSMIBlock(nn.Module):
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
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_m = x
        for blk in self.mamba_branch:
            x_m = blk(x_m)

        x_c = x
        for blk in self.cnn_branch:
            x_c = blk(x_c)

        return self.fuse(x_m + x_c)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class BaseAblationUNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base_c=32, block_factory=None):
        super().__init__()
        if block_factory is None:
            raise ValueError("block_factory is required")

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
        )

        self.encoder1 = block_factory(base_c)
        self.down1 = nn.Conv2d(base_c, base_c * 2, 3, stride=2, padding=1)

        self.encoder2 = block_factory(base_c * 2)
        self.down2 = nn.Conv2d(base_c * 2, base_c * 4, 3, stride=2, padding=1)

        self.encoder3 = block_factory(base_c * 4)
        self.down3 = nn.Conv2d(base_c * 4, base_c * 8, 3, stride=2, padding=1)

        self.bottleneck = VSSBlock(hidden_dim=base_c * 8, dropout=0.1)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleConv(base_c * 8 + base_c * 4, base_c * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleConv(base_c * 4 + base_c * 2, base_c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(base_c * 2 + base_c, base_c)

        self.final_conv = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.stem(x)

        f1 = self.encoder1(x1)
        d1 = self.down1(f1)

        f2 = self.encoder2(d1)
        d2 = self.down2(f2)

        f3 = self.encoder3(d2)
        d3 = self.down3(f3)

        bot = self.bottleneck(d3)

        u3 = self.up3(bot)
        dec3 = self.conv3(torch.cat([u3, f3], dim=1))

        u2 = self.up2(dec3)
        dec2 = self.conv2(torch.cat([u2, f2], dim=1))

        u1 = self.up1(dec2)
        dec1 = self.conv1(torch.cat([u1, f1], dim=1))

        return self.final_conv(dec1)


class MambaAblationOnly(BaseAblationUNet):
    """Ablation Row-1: Mamba only (no CNN, no CSMI, no edge branch)."""

    def __init__(self, in_ch=1, num_classes=4, base_c=32):
        super().__init__(
            in_ch=in_ch,
            num_classes=num_classes,
            base_c=base_c,
            block_factory=lambda dim: MambaOnlyBlock(dim, depth=1),
        )


class MambaAblationCNN(BaseAblationUNet):
    """Ablation Row-2: Mamba + CNN (no CSMI, no edge branch)."""

    def __init__(self, in_ch=1, num_classes=4, base_c=32):
        super().__init__(
            in_ch=in_ch,
            num_classes=num_classes,
            base_c=base_c,
            block_factory=lambda dim: MambaCNNNoCSMIBlock(dim, depth=1),
        )
