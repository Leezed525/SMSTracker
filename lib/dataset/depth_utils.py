"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/2 12:31
"""

import cv2
import numpy as np


def get_rgbd_frame(color_path, depth_path, dtype='rgbcolormap', depth_clip=False):
    ''' read RGB and depth images

        max_depth = 10 meter, in the most frames in CDTB and DepthTrack , the depth of target is smaller than 10 m
        When on CDTB and DepthTrack testing, we use this depth clip
    '''
    if color_path:
        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = None

    if depth_path:
        dp = cv2.imread(depth_path, -1)

        if depth_clip:
            max_depth = min(np.median(dp) * 3, 10000)
            dp[dp>max_depth] = max_depth
    else:
        dp = None

    if dtype == 'color':
        img = rgb

    elif dtype == 'raw_depth':
        img = dp

    elif dtype == 'colormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.applyColorMap(dp, cv2.COLORMAP_JET)

    elif dtype == '3xD':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.merge((dp, dp, dp))

    elif dtype == 'normalized_depth':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = np.asarray(dp, dtype=np.uint8)

    elif dtype == 'rgbcolormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
        img = cv2.merge((rgb, colormap))

    elif dtype == 'rgb3d':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        dp = cv2.merge((dp, dp, dp))
        img = cv2.merge((rgb, dp))

    elif dtype == 'rgbrgb':
        dp = cv2.cvtColor(dp, cv2.COLOR_BGR2RGB)
        img = cv2.merge((rgb, dp))

    else:
        print('No such dtype !!! ')
        img = None

    return img


def get_x_frame(color_path, depth_path, dtype='rgbcolormap', depth_clip=False):
    """
    read RGB and depth images  get_rgbd_frame

        max_depth = 10 meter, in the most frames in CDTB and DepthTrack , the depth of target is smaller than 10 m
        When on CDTB and DepthTrack testing, we use this depth clip
    :param color_path: rgb图像路径
    :param depth_path: depth图像路径
    :param dtype: 读取类型
    :param depth_clip: 是否裁剪深度
    :return:
    """
    if color_path:
        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = None

    if depth_path:
        dp = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_clip:
            max_depth = min(np.median(dp) * 3, 10000)
            dp[dp > max_depth] = max_depth
    else:
        dp = None

    if dtype == 'color':
        img = rgb
    elif dtype == 'raw_x':
        img = dp
    elif dtype == 'colormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
    elif dtype == '3x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        img = cv2.merge((dp, dp, dp))
    elif dtype == 'normalized_x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = np.asarray(dp, dtype=np.uint8)
    elif dtype == 'rgbcolormap':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)
        img = cv2.merge((rgb, colormap))

    elif dtype == 'rgb3x':
        dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dp = np.asarray(dp, dtype=np.uint8)
        dp = cv2.merge((dp, dp, dp))
        img = cv2.merge((rgb, dp))

    elif dtype == 'rgbrgb':
        dp = cv2.cvtColor(dp, cv2.COLOR_BGRA2BGR)
        img = cv2.merge((rgb, dp))
    else:
        print("No such dtype!!!")
        img = None
    return img
