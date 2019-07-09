# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:09:45 2019

@author: 37112
"""

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from lib.config import config as cfg

def load_annotation(index):
    
    #Load image and bounding boxes info from XML file 
    data_path = 'C:/Users/37112/Faster-RCNN-TensorFlow-Python3.5-master/data/VOCDevkit2007/VOC2007'
    filename = os.path.join(data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    if not False:
        # Exclude the samples labeled as difficult
        non_diff_objs = [
            obj for obj in objs if int(obj.find('difficult').text) == 0]
        # if len(non_diff_objs) != len(objs):
        #     print 'Removed {} difficult objects'.format(
        #         len(objs) - len(non_diff_objs))
        objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = []

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) 
        y1 = float(bbox.find('ymin').text) 
        x2 = float(bbox.find('xmax').text) 
        y2 = float(bbox.find('ymax').text) 
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes.append(obj.find('name').text.lower().strip())

    return boxes,gt_classes

#用来在图片中展示出bbox，包括物体框和类别。
def demo(boxes, gt_classes, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    #检测目标类，在图片中提议窗口  
    # Load the demo image
#    print("boxes: {},gt_classes:{}".format(boxes,gt_classes))
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(gt_classes)):
        ax.add_patch(
            plt.Rectangle((boxes[i][0], boxes[i][1]),
                          boxes[i][2] - boxes[i][0],
                          boxes[i][3] - boxes[i][1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(boxes[i][0], boxes[i][1] - 2,
                '{:s}'.format(gt_classes[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')  #不显示坐标尺寸
    plt.tight_layout()
#        plt.draw()


if __name__ == '__main__':
    
#    im_names = ['000004.jpg','000014.jpg','000024.jpg','000034.jpg','000044.jpg',
#                '000735.jpg','000865.jpg','000815.jpg','000795.jpg','000805.jpg']
    im_names = ['002188.jpg','002189.jpg','002190.jpg','002191.jpg']
#    im_names = ['000016.jpg']
    for im_name in im_names:
#        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        index = im_name.split('.')
        boxes,gt_classes = load_annotation(index[0])
        demo(boxes, gt_classes, im_name)

    plt.show()
