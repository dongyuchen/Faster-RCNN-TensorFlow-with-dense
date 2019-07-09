# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:08:18 2019

@author: 37112
"""
import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

if __name__ == '__main__':
    root_path = os.path.abspath('.')
    xml_path = os.path.abspath(os.path.join(root_path, 'data/VOCDevkit2007/VOC2007/Annotations/'))
    out_path = os.path.abspath(os.path.join(root_path, 'data/VOCDevkit2007/VOC2007/Annotations_txt/'))
    for i in range(300):
        file_name = os.path.join(xml_path, '00' + str(1892 + i) + '.xml')
        objects = parse_rec(file_name)
        out_file = os.path.join(out_path, '00'+ str(1892 + i) + '.txt')
        if not os.path.isfile(out_file):
            f = open(out_file, 'w')
        else:
            f = open(out_file, 'a')
        for i in range(len(objects)):
            bbox = ''
            for j in range(len(objects[i]['bbox'])):
                bbox += str(objects[i]['bbox'][j]) + ' '
            f.writelines((objects[i]['name'] + ' ' + bbox).strip() + '\n')
        f.close()
#        f.write((objects[i]['name'], objects[i]['bbox'][j] for j in range(len(objects[i]['bbox']))) for i in range(len(objects)))
        