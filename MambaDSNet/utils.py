import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.num_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.num_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.num_classes


def get_edge_gt(mask):
    """由分割标签生成二值边缘真值（形态学梯度：膨胀 - 腐蚀）。

    输入 mask:
        - [B, H, W]（整数类别标签），或
        - [B, 1, H, W]（二值标签）
    输出: edge [B, 1, H, W]，取值 0/1。
    基于 GPU max-pool 实现，无需 CPU 回传。
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    # 膨胀
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    # 腐蚀（取负 -> maxpool -> 取负）
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = torch.abs(dilated - eroded)
    return (edge > 0).float()


def bce_dice_loss(pred, target):
    """二值分割损失 = BCE + Dice。

    pred:   模型 logits，shape [B, 1, H, W]
    target: 二值标签，shape [B, 1, H, W]，取值 0/1
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    smooth = 1e-5
    prob = torch.sigmoid(pred)
    intersection = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - (2.0 * intersection + smooth) / (union + smooth)
    return bce + dice.mean()
