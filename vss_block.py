import torch
import torch.nn as nn
import torch.nn.functional as F

# mamba_ssm 依赖检查（SS2D 内部调用 Mamba）
try:
    import mamba_ssm  # noqa: F401
except ImportError:
    print("错误: 未找到 mamba_ssm，请先安装: pip install mamba-ssm")


class SS2D(nn.Module):
    """单向 Vision Mamba（已移除 4 向 Cross-Scan，仅保留单向扫描）。"""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 2D 深度卷积（等价于 Mamba 的一维因果卷积，在 2D 上实现）
        # 偶数核的 "same" padding 需非对称填充，故手动 pad 后再卷积
        pad_total = d_conv - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        self.pad = (pad_before, pad_after, pad_before, pad_after)  # (left, right, top, bottom)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=0,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 内部调用 mamba_ssm 的 selective_scan CUDA/CPU 算子
        from mamba_ssm import Mamba

        self.mamba = Mamba(
            d_model=self.d_inner,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # 将特征映射到输出通道，作为边界/方向门控
        self.dir_proj = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1, bias=True)

        # 序列长度不匹配时仅提示一次
        self._seq_warned = False

    def forward(self, x: torch.Tensor, **kwargs):
        # Input: [B, H, W, C]
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, _ = xz.chunk(2, dim=-1)  # [B, H, W, d_inner]

        x = x.permute(0, 3, 1, 2).contiguous()  # [B, d_inner, H, W]
        x = F.pad(x, self.pad, mode="reflect")
        x = self.act(self.conv2d(x))  # Depthwise Conv
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, d_inner]

        # 单向扫描：直接展平为序列进行 Mamba 处理
        seq = x.reshape(B, -1, self.d_inner).contiguous()
        seq = self.mamba(seq)

        # 序列长度变化时插值回 H*W
        L_expected = H * W
        L_out = seq.shape[1]
        if L_out != L_expected:
            if not self._seq_warned:
                print(f"[SS2D] seq length mismatch: expected {L_expected}, got {L_out}. Interpolating once.")
                self._seq_warned = True
            seq = seq.transpose(1, 2)  # [B, C, L]
            seq = F.interpolate(seq, size=L_expected, mode="linear", align_corners=False)
            seq = seq.transpose(1, 2).contiguous()

        # 还原形状
        y = seq.view(B, H, W, self.d_inner)
        
        y_norm = self.out_norm(y)
        y_out = self.out_proj(y_norm)  # [B, H, W, d_model]

        if self.dropout is not None:
            y_out = self.dropout(y_out)

        # 用特征本身计算 dir_att，供下游 CSMI Fusion 使用
        dir_att = torch.sigmoid(self.dir_proj(y.permute(0, 3, 1, 2)))
        return y_out, dir_att


# 简化版 VSSBlock（内部调用 SS2D）
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            dropout: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)

        self.ss2d = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, input: torch.Tensor):
        # Input: [B, C, H, W] -> SS2D expects [B, H, W, C]
        B, C, H, W = input.shape
        x = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(x)

        x, dir_att = self.ss2d(x)  # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2).contiguous()

        # 保存方向门控供上层融合使用
        self.last_dir_att = dir_att

        return input + self.dropout(x)