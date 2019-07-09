# tf-faster-rcnn
This is the branch to improve dBeker's job, adding the DenseNet as the basic net(the dense put in the lib/nets/dense) which including 121 and 169 two types layer. 

Note that: you can clone dBeker's branch and download my densenet, train.py, demo.py, config to run densenet with faster rcnn. Because in my branch it miss the coco filedir in data, you can compare it with dBeker's
All the classes in my program just for my own data, you need to change that to train your own datasets.
To change the classes in the following two places:
1. lib/datasets/pascal_voc.py
2. demo
# How To Use This Branch
To run the program can follow the step of https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3

When you want to use densenet to train the model just change the lib.config.config 'network' as the dense
 
pre-trained model(densenet121 and densenet161) you can download from https://drive.google.com/file/d/0B_fUSpodN0t0TDB5Ti1PeTZMM2c/view and https://drive.google.com/file/d/0B_fUSpodN0t0eW1sVk1aeWREaDA/view

I also add the program which can compute mAP and iou(compute_mAP_IoU.py). To run this program, it needs some pre-processing to the data.
xml2txt.py: to get the annotation from xml file and save as txt
textwrite.py: to save the predict results as txt file.
demo-show-gt.py: can show the ground truth bboxes for the picture you want to see.
