"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/19 18:10
"""
import torch
import torchvision.transforms as transforms
import math
import torch.nn.functional as F
import cv2 as cv
import numpy as np

from lib.common.tensor import TensorDict


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        else:
            mask_crop_padded = F.interpolate(mask_crop_padded[None, None],
                                             (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]  # 上采样函数
            return im_crop_padded, resize_factor, att_mask, mask_crop_padded
    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """
    For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates
    :param frames: list of frames
    :param box_extract: list of boxes of same length as frames. The crops are extracted using anno_extract
    :param box_gt: list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
    :param search_area_factor: The area of the extracted crop is search_area_factor^2 times box_extract area
    :param output_sz: The size to which the extracted crops are resized
    :param masks:
    :return:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
    """
    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """
    基础数据处理类
    主要负责将数据集中的数据进行处理，包括数据增强，数据转换等，最后转换成tensor

    """

    def __init__(self, transform=transforms.ToTensor(), template_transformer=None, search_transformer=None, joint_transformer=None):
        self.transform = {
            'template': transform if template_transformer is None else template_transformer,
            'search': transform if search_transformer is None else search_transformer,
            'joint': joint_transformer
        }

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class ViPTProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, transform=transforms.ToTensor(),
                 template_transform=None, search_transform=None, joint_transform=None, mode='pair'):
        """

        :param search_area_factor:  The size of the search region  relative to the target size.
        :param output_sz: An integer, denoting the size to which the search region is resized. The search region is always square.
        :param center_jitter_factor: A dict containing the amount of jittering to be applied to the target center before extracting the search region. See _get_jittered_box for how the jittering is done.
        :param scale_jitter_factor: A dict containing the amount of jittering to be applied to the target size before extracting the search region. See _get_jittered_box for how the jittering is done.
        :param transform: see BaseProcessing
        :param template_transformer: see BaseProcessing
        :param search_transformer: see BaseProcessing
        :param joint_transformer: see BaseProcessing
        :param mode: Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(transform, template_transform, search_transform, joint_transform)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """
        Jitter the input box
        :param box: input bounding box
        :param mode: string 'template' or 'search' indicating template or search data
        :return: torch.Tensor - jittered box
        """
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'])
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'],
                                                                                 new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, "In pair mode,num train/test frames must be 1"

            # Add a uniform noise to the center pos,RGB and X modalities are aligned
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Check whether data is valid .Avoid too small bounding boxes stack (Ns,4)
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                return data
            crops, boxes, _, _ = jittered_center_crop(data[s + '_images'], jittered_anno,
                                                      data[s + '_anno'], self.search_area_factor[s],
                                                      self.output_sz[s])

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)
        data['valid'] = True

        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
