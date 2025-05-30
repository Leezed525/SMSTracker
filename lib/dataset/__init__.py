"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/26 18:49
"""
from lib.dataset.DepthTrack import DepthTrack
from lib.config.cfg_loader import env_setting
from lib.dataset.LasHeR import LasHeR
from lib.dataset.VisEvent import VisEvent
from lib.config.cfg_loader import CfgLoader


def names_to_datasets(name_list: list, cfg: CfgLoader, image_loader):
    # cfg = env_setting(cfg_name=None)
    datasets = []
    valid_dataset_name = ["LasHeR_all", "LasHeR_train", "LasHeR_val", "DepthTrack_train", "DepthTrack_val", "VisEvent_train", "VisEvent_val"]
    for name in name_list:
        assert name in valid_dataset_name, "Invalid dataset name:{}".format(name)
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(root=cfg.dataset.DepthTrack.train.dir, dtype='rgbcolormap', split='train'))
        if name == 'DepthTrack_val':
            datasets.append(DepthTrack(root=cfg.dataset.DepthTrack.val.dir, dtype='rgbcolormap', split='val'))
        if name == 'LasHeR_all':
            datasets.append(LasHeR(root=cfg.dataset.LasHeR.train.dir, split='all', dtype='rgbrgb'))
        if name == 'LasHeR_train':
            datasets.append(LasHeR(root=cfg.dataset.LasHeR.train.dir, split='train', dtype='rgbrgb'))
        if name == 'LasHeR_val':
            datasets.append(LasHeR(root=cfg.dataset.LasHeR.val.dir, split='val', dtype='rgbrgb'))
        if name == "VisEvent_train":
            datasets.append(VisEvent(root=cfg.dataset.VisEvent.train.dir, dtype='rgbrgb', split='train'))
        if name == "VisEvent_val":
            datasets.append(VisEvent(root=cfg.dataset.VisEvent.train.dir, dtype='rgbrgb', split='val'))

    return datasets
