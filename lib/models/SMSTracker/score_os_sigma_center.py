"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/16 20:26
"""
import torch
import torch.nn as nn
from lib.config.cfg_loader import CfgLoader

from lib.models.backbone.vit_sigma import vit_base_patch16_224
from timm.models.layers import to_2tuple
from lib.models.head.center_predictor_origin import build_box_head


class ScoreOSCENTER(nn.Module):
    def __init__(self, backbone, box_head, cfg: CfgLoader = None, head_type='CENTER'):
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type

        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)

        self.cfg = cfg

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn)

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # print("cat_feature",cat_feature.shape)
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # print("opt_feat", opt_feat.shape)

        if self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            # print("outputs_coord", outputs_coord.shape)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_score_os_sigma_center(cfg, training=True):
    backbone = vit_base_patch16_224(pretrained=False,
                                       drop_path_rate=cfg.train.drop_path_rate,
                                       search_size=to_2tuple(cfg.data.search.size),
                                       template_size=to_2tuple(cfg.data.template.size),
                                       new_patch_size=cfg.model.backbone.stride)
    # backbone = vit_base_patch16_224_ce(pretrained=False,
    #                                    drop_path_rate=cfg.train.drop_path_rate,
    #                                    ce_loc=cfg.model.backbone.ce_loc,
    #                                    ce_keep_ratio=cfg.model.backbone.ce_keep_ratio,
    #                                    search_size=to_2tuple(cfg.data.search.size),
    #                                    template_size=to_2tuple(cfg.data.template.size),
    #                                    new_patch_size=cfg.model.backbone.stride)
    # from lib.models.backbone.vit_ce_prompt import vit_base_patch16_224_ce_prompt
    # backbone = vit_base_patch16_224_ce_prompt(pretrained=False, drop_path_rate=cfg.train.drop_path_rate,
    #                                           ce_loc=cfg.model.backbone.ce_loc,
    #                                           ce_keep_ratio=cfg.model.backbone.ce_keep_ratio,
    #                                           search_size=to_2tuple(cfg.data.search.size),
    #                                           template_size=to_2tuple(cfg.data.template.size),
    #                                           new_patch_size=cfg.model.backbone.stride,
    #                                           prompt_type="vipt_deep"
    #                                           )
    hidden_dim = backbone.embed_dim

    box_head = build_box_head(cfg, hidden_dim)

    model = ScoreOSCENTER(
        backbone,
        box_head,
    )
    return model
