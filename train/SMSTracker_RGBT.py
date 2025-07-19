"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/18 19:47
"""
import os
import sys

sys.path.append(os.getcwd())
from lib.actor.SMSTracker import SMSTrackerActor
from lib.models.SMSTracker.score_os_sigma_center import build_score_os_sigma_center
from lib.trainer.LeeNet_trainer import LeeNetTrainer
from lib.utils.base_funtion import build_dataloaders, get_optimizer_scheduler
from lib.config.cfg_loader import env_setting
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.backends.cudnn
import argparse
import torch.distributed as dist
import random
import numpy as np


def init_seeds(seed):
    print("设置随机种子为：", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run():
    dist.init_process_group(backend='nccl')
    cfg = env_setting(cfg_name="plscore_os_sigma_RGBT.yaml")
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    seed = 3407

    # 多卡配置
    if local_rank != -1:
        print("使用多卡训练")
        torch.cuda.set_device(local_rank)
        init_seeds(seed + local_rank)
    else:
        device = cfg.train.device
        print("使用" + device + "训练")
        torch.cuda.set_device(cfg.train.device)
        init_seeds(seed)

    torch.backends.cudnn.benchmark = True

    loader_train, loader_val = build_dataloaders(cfg, world_size, local_rank)

    net = build_score_os_sigma_center(cfg)

    # 导入预训练权重
    pretrained = "/media/star/data/Leezed/workspace/SMSTracker/pretrained/OSTrack_ep0300.pth.tar"
    checkpoint = torch.load(pretrained)['net']

    model_dict = net.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict, strict=False)
    print("导入预训练权重成功")

    # 导入预训练权重结束

    net.cuda()
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)

    focal_loss = FocalLoss()
    objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
    loss_weight = {'giou': cfg.train.GIOU_weight, 'l1': cfg.train.L1_weight, 'focal': 1., 'cls': 1.0}
    actor = SMSTrackerActor(net=net, objective=objective, loss_weight=loss_weight, cfg=cfg)

    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)


    trainer = LeeNetTrainer(actor=actor, loaders=[loader_train, loader_val], optimizer=optimizer, lr_scheduler=lr_scheduler, cfg=cfg, rank=local_rank)
    print("开始训练")
    trainer.train(cfg.train.epoch, load_latest=True)


if __name__ == '__main__':
    run()
