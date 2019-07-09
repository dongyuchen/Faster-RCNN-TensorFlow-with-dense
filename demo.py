#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#引入argparse， 它python用于解析命令行参数和选项的
#标准模块，用于解析命令行参数
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
#nms 进行非极大值抑制
from lib.utils.nms_wrapper import nms
#im_detect 生成RPN候选框
from lib.utils.test import im_detect
from lib.nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.nets.dense import dense
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'specularity', 'saturation', 'artifact', 'blur',
           'contrast', 'bubbles', 'instrument')

NETS = {'vgg16': ('vgg16.ckpt',),'res101': ('res.ckpt',),'dense': ('dense.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

#用来使得检测结果可视化，即在图片中展示出检测结果，包括物体框和类别以及得分。
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    #选取候选框score大于阈值的dets
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    # python-opencv 中读取图片默认保存为[w,h,channel]
    # 其中 channel：BGR 存储，而画图时，需要按RGB格式，因此此处作转换。
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')#如果是‘equal’, 并且extent为None,坐标系长宽比将匹配图像
    for i in inds:
        #从dets中取出 bbox, score
        bbox = dets[i, :4]#坐标位置（Xmin,Ymin,Xmax,Ymax）
        score = dets[i, -1]#置信度得分  
        #根据起始点坐标以及w,h 画出矩形框
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),#bbox[0]:x, bbox[1]:y, bbox[2]:x+w, bbox[3]:y+h
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')  #不显示坐标尺寸
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    #检测目标类，在图片中提议窗口  
    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic() #返回开始时间
    scores, boxes = im_detect(sess, net, im)#检测，返回得分和区域所在位置
    timer.toc()#返回平均时间
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    
    # Visualize detections for each class
    #score 阈值，最后画出候选框时需要，>thresh才会被画出
    CONF_THRESH = 0.1
    #非极大值抑制的阈值，剔除重复候选框
    NMS_THRESH = 0.4
    #利用enumerate函数，获得CLASSES中类别的下标cls_ind和类别名cls
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        #将bbox,score 一起存入dets,按列顺序把数组给堆叠
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #进行非极大值抑制，得到抑制后的 dets
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #画框
#        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        
        """Draw detected bounding boxes."""
        #选取候选框score大于阈值的dets
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
#            return
            continue

        for i in inds:
#            print("外层循环{}次，进入内层循环".format(cls_ind))
            bbox = dets[i, :4]#坐标位置（Xmin,Ymin,Xmax,Ymax）
            score = dets[i, -1]#置信度得分  
            #根据起始点坐标以及w,h 画出矩形框
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(cls, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
        plt.axis('off')  #不显示坐标尺寸
        plt.tight_layout()
        plt.draw()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101 dense]',
                        choices=NETS.keys(), default='dense')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()#模式设置

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    #设置训练好的模式的路径
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = False

    # init session
#    sess = tf.Session(config=tfconfig)
    sess = tf.Session()
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
         net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'dense':
        net = dense(batch_size=1, num_layers=121)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 8,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    im_names = ['002188.jpg','002189.jpg','002190.jpg','002191.jpg']

#    im_names = ['D_EAD2019_0.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

    plt.show()
