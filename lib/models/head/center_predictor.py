"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/5/6 19:12
"""
import torch.nn as nn
import torch
from lib.models.layer.Conv import Conv


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        self.gen_feat_conv = Conv(cin=inplanes, cout=inplanes, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))

        # corner predict
        self.conv1_ctr = Conv(cin=inplanes, cout=channel, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv2_ctr = Conv(cin=channel, cout=channel // 2, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv3_ctr = Conv(cin=channel // 2, cout=channel // 4, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv4_ctr = Conv(cin=channel // 4, cout=channel // 8, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = Conv(cin=inplanes, cout=channel, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv2_offset = Conv(cin=channel, cout=channel // 2, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv3_offset = Conv(cin=channel // 2, cout=channel // 4, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv4_offset = Conv(cin=channel // 4, cout=channel // 8, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = Conv(cin=inplanes, cout=channel, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv2_size = Conv(cin=channel, cout=channel // 2, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv3_size = Conv(cin=channel // 2, cout=channel // 4, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv4_size = Conv(cin=channel // 4, cout=channel // 8, kernel_size=3, stride=1, activation=nn.ReLU(inplace=True))
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        x = self.gen_feat_conv(x)
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset


def build_box_head(cfg, hidden_dim):
    stride = cfg.model.backbone.stride

    in_channel = hidden_dim
    out_channel = cfg.model.head.num_channels
    feat_sz = int(cfg.data.search.size / stride)
    center_head = CenterPredictor(inplanes=in_channel, channel=out_channel, feat_sz=feat_sz, stride=stride)
    return center_head
