"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/14 22:17
"""
import torch.nn as nn
# from lib.models.layer.RMT import RetBlock, RelPos2d
from lib.models.layer.Conv import Conv
import torch
from lib.models.layer.attn import Attention


class ScoreLayerUseConv(nn.Module):
    # 废案，这个方法梯度无法回传
    def __init__(self, threshold1=0.33, threshold2=0.66, embed_dim=768, num_heads=12, ffn_dim=96, initial_value=1, heads_range=3):
        super().__init__()
        # self.rel_pos = RelPos2d(embed_dim=embed_dim, num_heads=num_heads, initial_value=initial_value, heads_range=heads_range)
        # self.ret_block = RetBlock(retention='whole', embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim)
        self.embed_dim = embed_dim
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.conv1 = Conv(embed_dim, embed_dim // 2, 5, 1)
        self.conv2 = Conv(embed_dim // 2, 1, 3, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (B, H, W, C)
        # relpos = self.rel_pos((H, W))
        # x = self.ret_block(x, retention_rel_pos=relpos)
        # x = x.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)

        # x shape: (B,C,H,W)
        mask = self.softmax(self.conv2(self.conv1(x)))

        positive_mask = (mask < 0.33).float()
        uncertain_mask = ((mask >= 0.33) & (mask < 0.66)).float()
        negative_mask = (mask >= 0.66).float()
        return positive_mask, uncertain_mask, negative_mask


class PLScoreLayerUseConv(nn.Module):
    def __init__(self, threshold1=0.33, threshold2=0.66, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv1 = Conv(embed_dim, embed_dim // 2, 5, 1)
        self.conv2 = Conv(embed_dim // 2, 1, 3, 1)

        self.sig = nn.Sigmoid()

        self.avg = nn.AdaptiveAvgPool2d((64, 64))
        self.confident_conv1 = Conv(embed_dim * 2, embed_dim // 2, 8, 8, 0)
        self.confident_conv2 = Conv(embed_dim // 2, 1, 8, 1, 0)

    def forward(self, x_rgb, x_modal):
        # 使用clone来避免梯度回传报错的问题
        x_rgb_clone = x_rgb.clone()
        x_modal_clone = x_modal.clone()

        # x shape: (B,C,H,W)
        positive_mask = self.sig(self.conv2(self.conv1(x_rgb_clone)))  # (B,1,H,W)
        negative_mask = 1 - positive_mask

        # 根据x_rgb 和 x_modal 来生成整体rgb图像的可信度，输出一个在0.5 - 0.9之间的值
        x_tmp = torch.cat((x_rgb_clone, x_modal_clone), dim=1)
        x_tmp = self.avg(x_tmp)
        x_tmp = self.sig(self.confident_conv2(self.confident_conv1(x_tmp)))
        value = x_tmp * 0.4 + 0.5
        x = positive_mask * (value * x_rgb_clone + (1 - value) * x_modal_clone) + (value * x_rgb_clone + value * x_modal_clone) + negative_mask * (
                (1 - value) * x_rgb_clone + value * x_modal_clone)  # 20240722 0:08把中间的0.5换成了value

        return x


class ScoreAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_drop=0.1, qkv_bias=False, drop=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.merge = nn.MaxPool1d(kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x, x_modal):
        # x shape: (B, C ,D)
        B, C, D = x.shape
        x_score = self.attn(x)
        x_modal_score = self.attn(x_modal)

        x_score = x_score.flatten(1)  # (B, C, D) -> (B, C * D)
        x_modal_score = x_modal_score.flatten(1)

        x_score = x_score.unsqueeze(1)
        x_modal_score = x_modal_score.unsqueeze(1)

        x = torch.cat((x_score, x_modal_score), dim=1)
        x = self.merge(x).reshape(B, C, D)
        x = self.norm(x)

        return x
