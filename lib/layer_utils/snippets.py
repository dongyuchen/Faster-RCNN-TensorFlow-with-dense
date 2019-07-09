# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.layer_utils.generate_anchors import generate_anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
    """
    """生成anchor的预处理方法，generate_anchors方法就是直接产生各种大小的anchor box，generate_anchors_pre方法
     是把每一个anchor box对应到原图上
      feat_stride: 经过VGG或者ZF后特征图相对于原图的在长或者宽上的缩放倍数，也就是说height和width对应于特征图长宽
      anchor_scales：anchor尺寸
      anchor_ratios: anchor长宽比
    """

    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0] #A=9
#    print("A:",A)
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
#    print("K:",K)
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length
