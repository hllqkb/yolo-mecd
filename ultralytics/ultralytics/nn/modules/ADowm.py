import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class ADown(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        assert c1 % 2 == 0
        self.cv1 = Conv(c1 // 2, c2 // 2, 3, 2, 1)  # Conv 下采样
        self.cv2 = Conv(c1 // 2, c2 // 2, 1, 1, 0)  # 通道压缩
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化下采样

    def forward(self, x):
        assert x.shape[1] % 2 == 0, "ADown: 输入通道数必须是2的倍数"
        x1, x2 = x.chunk(2, dim=1)  # 通道切分
        x1 = self.cv1(x1)           # 3x3 conv, stride=2
        x2 = self.pool(x2)          # 2x2 maxpool, stride=2
        x2 = self.cv2(x2)           # 1x1 conv
        assert x1.shape[-2:] == x2.shape[-2:], f"ADown: x1={x1.shape}, x2={x2.shape}"  # 尺寸匹配确认
        return torch.cat((x1, x2), dim=1)
