"""
cfg生成器
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/18 13:25
"""
import os.path

from lib.utils.YamlUtil import YamlUtil


def save_cfg(cfg: YamlUtil):
    """
    保存配置文件
    :param cfg:
    :return:
    """
    cfg.write_yaml()


def default_cfg_generate(cfg_name, workspeace_dir, dataset_dir):
    """
    生成默认配置
    :param cfg_name: 配置文件名
    :param workspeace_dir:  工作空间路径
    :param dataset_dir:  数据集路径
    :return:
    """
    cfg = YamlUtil(os.path.join(workspeace_dir, "configs"), cfg_name)
    cfg.add_root_key('model')
    cfg.add_root_key('dataset')
    cfg.add_root_key('train')
    cfg.add_root_key('data')
    dataset_cfg_generate(cfg, dataset_dir)


def dataset_cfg_generate(cfg: YamlUtil, dataset_dir):
    """
    生成数据集相关配置
    :param cfg: cfg
    :param dataset_dir: 数据集路径
    :return:
    """
    cfg.add_key('dataset.DepthTrack')
    cfg.add_key('dataset.DepthTrack.train')
    cfg.add_key('dataset.DepthTrack.train.dir', dataset_dir + '\\DepthTrack\\train')
    cfg.add_key('dataset.DepthTrack.val')
    cfg.add_key('dataset.DepthTrack.val.dir', dataset_dir + '\\DepthTrack\\train')

    cfg.add_key('dataset.LasHeR')
    cfg.add_key('dataset.LasHeR.train')
    cfg.add_key('dataset.LasHeR.train.dir', dataset_dir + '\\LasHeR\\train')
    cfg.add_key('dataset.LasHeR.val')
    cfg.add_key('dataset.LasHeR.val.dir', dataset_dir + '\\LasHeR\\train')

    cfg.add_key('dataset.VisEvent')
    cfg.add_key('dataset.VisEvent.train')
    cfg.add_key('dataset.VisEvent.train.dir', dataset_dir + '\\VisEvent\\train')
    cfg.add_key('dataset.VisEvent.val')
    cfg.add_key('dataset.VisEvent.val.dir', dataset_dir + '\\VisEvent\\val')

    save_cfg(cfg)
