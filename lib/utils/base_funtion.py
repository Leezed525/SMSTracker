"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/13 15:51
"""

from lib.config.cfg_loader import CfgLoader
import lib.data.transforms as tfm
from lib.dataset import names_to_datasets
from lib.data import processing, sampler, image_loader, loader
import torch
from torch.utils.data.distributed import DistributedSampler

from lib.utils.multigpu import is_main_process


def build_dataloaders(cfg: CfgLoader, world_size=1, local_rank=-1):
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.data.mean, std=cfg.data.std))
    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.data.mean, std=cfg.data.std))

    output_size = {
        'template': cfg.data.template.size,
        'search': cfg.data.search.size
    }
    search_area_factor = {
        'template': cfg.data.template.factor,
        'search': cfg.data.search.factor
    }
    center_jitter_factor = {
        'template': cfg.data.template.center_jitter,
        'search': cfg.data.search.center_jitter
    }
    scale_jitter_factor = {
        'template': cfg.data.template.scale_jitter,
        'search': cfg.data.search.scale_jitter
    }

    data_processing_train = processing.ViPTProcessing(search_area_factor=search_area_factor,
                                                      output_sz=output_size,
                                                      center_jitter_factor=center_jitter_factor,
                                                      scale_jitter_factor=scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    data_processing_val = processing.ViPTProcessing(search_area_factor=search_area_factor,
                                                    output_sz=output_size,
                                                    center_jitter_factor=center_jitter_factor,
                                                    scale_jitter_factor=scale_jitter_factor,
                                                    mode='sequence',
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    sampler_mode = cfg.data.sampler_mode
    train_cls = cfg.train.train_cls

    dataset_train = sampler.TrackingSampler(datasets=names_to_datasets(cfg.data.train.datasets_name, cfg, image_loader.opencv_loader),
                                            p_datasets=cfg.data.train.datasets_ratio,
                                            samples_per_epoch=cfg.data.train.sample_per_epoch,
                                            max_gap=cfg.data.max_sample_interval,
                                            num_search_frames=cfg.data.search.number,
                                            num_template_frames=cfg.data.template.number,
                                            processing=data_processing_train,
                                            frame_sample_mode=sampler_mode,
                                            train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train, rank=local_rank) if local_rank != -1 else None
    shuffle = False if local_rank != -1 else True

    loader_train = loader.LTRLoader(name='train',
                                    dataset=dataset_train,
                                    training=True,
                                    batch_size=cfg.train.batch_size,
                                    shuffle=shuffle,
                                    num_workers=cfg.train.num_worker,
                                    drop_last=True,
                                    stack_dim=1,
                                    sampler=train_sampler)

    if len(cfg.data.val.datasets_name) == 0:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names_to_datasets(cfg.data.val.datasets_name, cfg, image_loader.opencv_loader),
                                              p_datasets=cfg.data.val.datasets_ratio,
                                              samples_per_epoch=cfg.data.val.sample_per_epoch,
                                              max_gap=cfg.data.max_sample_interval,
                                              num_search_frames=cfg.data.search.number,
                                              num_template_frames=cfg.data.template.number,
                                              processing=data_processing_val,
                                              frame_sample_mode=sampler_mode,
                                              train_cls=train_cls)
        val_sampler = DistributedSampler(dataset_val, rank=local_rank) if local_rank != -1 else None
        loader_val = loader.LTRLoader(name='val',
                                      dataset=dataset_val,
                                      training=False,
                                      batch_size=cfg.train.batch_size,
                                      num_workers=cfg.train.num_worker,
                                      drop_last=True,
                                      stack_dim=1,
                                      sampler=val_sampler,
                                      epoch_interval=cfg.train.val_epoch_interval)
    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    if 'plScore_OS_sigma' in cfg.train.specifical_model_name:
        # print("only train sigma and plscore parameters")
        train_params = ["cross_mamba", "channel_attn_mamba", 'score', "sigma_norm", "patch_embed_modal"]
        fine_tune_params = ["backbone.norm", "backbone.blocks"]
        # fine_tune_params = ["backbone.norm"]
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if any(nd in n for nd in train_params) and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if any(nd in n for nd in fine_tune_params) and p.requires_grad],
                "lr": cfg.train.lr * cfg.train.backbone_multiplier
            },
        ]
        print("Learnable parameters are shown below.")
        for n, p in net.named_parameters():
            if not any(nd in n for nd in train_params) and not any(nd in n for nd in fine_tune_params):
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters()]},
        ]
    # train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", "")
    # if 'vipt' in train_type:
    #     # print("Only training prompt parameters. They are: ")
    #     param_dicts = [
    #         {"params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad]}
    #     ]
    #     for n, p in net.named_parameters():
    #         if "prompt" not in n:
    #             p.requires_grad = False
    #         # else:
    #         #     print(n)
    # else:
    #     param_dicts = [
    #         {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
    #         {
    #             "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
    #             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
    #         },
    #     ]
    #     if is_main_process():
    #         print("Learnable parameters are shown below.")
    #         for n, p in net.named_parameters():
    #             if p.requires_grad:
    #                 print(n)
    if cfg.train.optimizer == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.train.scheduler.type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.train.lr_drop_epoch)
    else:
        raise ValueError("Unsupported Scheduler")
    # elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                         milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
    #                                                         gamma=cfg.TRAIN.SCHEDULER.GAMMA)

    return optimizer, lr_scheduler
