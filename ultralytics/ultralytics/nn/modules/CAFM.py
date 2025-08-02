# ultralytics/nn/modules/cafm.py
import torch
import torch.nn as nn
from einops import rearrange

class CAFM(nn.Module):
    def __init__(self, c1, c2, num_heads=8, bias=False):
        super().__init__()
        # 确保 c1 能被 num_heads 整除
        assert c1 % num_heads == 0, f"c1 ({c1}) must be divisible by num_heads ({num_heads})"
        self.dim = c1
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 3D卷积部分
        self.qkv = nn.Conv3d(c1, c1*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(c1*3, c1*3, kernel_size=(3,3,3), 
                                   stride=1, padding=1, groups=c1*3, bias=bias)
        self.project_out = nn.Conv3d(c1, c2, kernel_size=(1,1,1), bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)
        groups = max(1, c1 // num_heads)  # 确保至少为1
        out_channels = 9 * groups
        # 深度可分离卷积
        self.dep_conv = nn.Conv3d(out_channels, c2, 
                                 kernel_size=(3,3,3), bias=True,
                                 groups=groups, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)  # [B,C,1,H,W]
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)  # [B,3C,H,W]
        
        # 局部特征处理
        f_conv = qkv.permute(0,2,3,1) 
        f_all = qkv.reshape(b, h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3) 
        f_all = self.fc(f_all.unsqueeze(2)).squeeze(2)
        
        # 局部卷积分支
        f_conv = f_all.permute(0,3,1,2).reshape(b, 9*c//self.num_heads, h, w)
        out_conv = self.dep_conv(f_conv.unsqueeze(2)).squeeze(2)
        
        # 全局注意力分支
        q, k, v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = (attn.softmax(dim=-1) @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', 
                       head=self.num_heads, h=h, w=w)
        out = self.project_out(out.unsqueeze(2)).squeeze(2)
        
        return out + out_conv