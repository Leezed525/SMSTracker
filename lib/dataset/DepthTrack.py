"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/26 21:36
"""
import os

import numpy as np
import pandas
import torch

from lib.dataset.base_video_dataset import BaseVideoDataset
from lib.data.image_loader import jpeg4py_loader_w_failsafe
from lib.config.cfg_loader import env_setting
from lib.dataset.depth_utils import get_x_frame
from collections import OrderedDict


class DepthTrack(BaseVideoDataset):
    def __init__(self, root=None, dtype='rgbcolormap', split='train', image_loader=jpeg4py_loader_w_failsafe, cfg_name=None):
        root = env_setting(cfg_name).dataset.DepthTrack[split]['dir'] if root is None else root
        super(DepthTrack, self).__init__('DepthTrack', root, image_loader)
        self.dtype = dtype
        self.split = split
        self.sequence_list = self._build_sequence_list()
        self.seq_per_class, self.class_list = self._build_class_list()
        self.class_list.sort()
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

    def _build_sequence_list(self):
        # ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        ltr_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(ltr_path, 'data_specs', 'depthtrack_%s.txt' % self.split)
        sequence_list = pandas.read_csv(file_path, header=None).squeeze().values.tolist()
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        class_list = []
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('_')[0]

            if class_name not in class_list:
                class_list.append(class_name)

            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class, class_list

    def get_name(self):
        return "DepthTrack"

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequence_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        '''
        if the box is too small, it will be ignored
        '''
        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (bbox[:, 2] > 10.0) & (bbox[:, 3] > 10.0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        """
        返回某个seq某一帧的路径 (包含 color depth)
        :param seq_path:  sequence path
        :param frame_id:  frame id
        :return: color image path, depth image path
        """
        return os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id + 1)), os.path.join(seq_path, 'depth', '{:08}.png'.format(
            frame_id + 1))  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        color_path, depth_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(color_path, depth_path, self.dtype, depth_clip=True)

        return img

    def _get_class(self, seq_path):
        return self.split

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]

        frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motions_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta


