"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/26 21:22
"""

import torch.utils.data
from lib.data.image_loader import jpeg4py_loader_w_failsafe


class BaseVideoDataset(torch.utils.data.Dataset):
    def __init__(self, name, root, image_loader=jpeg4py_loader_w_failsafe):
        self.name = name
        self.root = root
        self.image_loader = image_loader

        self.sequence_list = []  # Contains the list of sequences.
        self.class_list = []

    def get_name(self):
        """
        Name of the dataset
        :return: string - Name of the dataset
        """
        raise NotImplementedError

    def get_num_sequences(self):
        """
        Number of sequences in a dataset
        :return: int - number of sequences in the dataset
        """
        return len(self.sequence_list)

    def is_video_sequence(self):
        """
        Returns whether the dataset is a video dataset or an image dataset
        :return: bool - True if a video dataset
        """
        return True

    def is_synthetic_video_dataset(self):
        """
        Returns whether the dataset contains real videos or synthetic
        :return: bool - True if a video dataset
        """
        return False

    def __len__(self):
        raise self.get_num_sequences()

    def __getitem__(self, idx):
        """
        Not to be used! Check get_frames() instead.
        :param idx:
        :return:
        """
        raise None

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_class_list(self):
        return self.class_list

    def get_sequence_in_class(self, class_name):
        raise NotImplementedError

    def has_segmentation_info(self):
        return False

    def get_sequence_info(self, seq_id):
        """
        return information about a particular sequences
        :param seq_id:  index of the sequence
        :return:  Dict
        """
        raise NotImplementedError

    def get_frames(self, seq_id, frame_ids, anno=None):
        """
        Get frames from a sequence
        :param seq_id: index of the sequence
        :param frame_ids: list of frame indices
        :param anno: list of annotations
        :return:
            list of frames corresponding to the frame_ids
            list of dicts for each frame
            dict with sequence information eg. class, occlusion, etc.

        """
        raise NotImplementedError
