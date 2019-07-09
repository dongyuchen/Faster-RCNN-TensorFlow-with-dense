# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:30:12 2019

@author: 37112
"""

# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import numpy as np

from lib.nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from lib.config import config as cfg

def densenet_arg_scope(is_training=True,
                       weight_decay=1e-4,
                       batch_norm_decay=0.99,
                       batch_norm_epsilon=1.1e-5,
                       data_format='NHWC'):
    with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.avg_pool2d, 
                         slim.max_pool2d, _conv_block, _global_avg_pool2d],
                      data_format=data_format):
        with slim.arg_scope([slim.conv2d],
                     weights_regularizer=slim.l2_regularizer(weight_decay),
                     activation_fn=None,
                     biases_initializer=None):
            with slim.arg_scope([slim.batch_norm],
                          scale=True,
                          decay=batch_norm_decay,
                          epsilon=batch_norm_epsilon) as scope:
                return scope

@slim.add_arg_scope
def _global_avg_pool2d(inputs, data_format='NHWC', scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'xx', [inputs]) as sc:
        axis = [1, 2] if data_format == 'NHWC' else [2, 3]
        net = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
        return net

@slim.add_arg_scope
def _conv(inputs, num_filters, kernel_size, stride=1, dropout_rate=None,
          scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'xx', [inputs]) as sc:
        net = slim.batch_norm(inputs)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, num_filters, kernel_size)

        if dropout_rate:
            net = tf.nn.dropout(net)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net

@slim.add_arg_scope
def _conv_block(inputs, num_filters, data_format='NHWC', scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'conv_blockx', [inputs]) as sc:
        net = inputs
        net = _conv(net, num_filters * 4, 1, scope='x1')
        net = _conv(net, num_filters, 3, scope='x2')
        if data_format == 'NHWC':
            net = tf.concat([inputs, net], axis=3)
        else: # "NCHW"
            net = tf.concat([inputs, net], axis=1)

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net

@slim.add_arg_scope
def _dense_block(inputs, convs_each_num_layers, num_filters, growth_rate,
                 grow_num_filters=True, scope=None, outputs_collections=None):

    with tf.variable_scope(scope, 'dense_blockx', [inputs]) as sc:
        net = inputs
        for i in range(convs_each_num_layers):
            branch = i + 1
            net = _conv_block(net, growth_rate, scope='conv_block'+str(branch))

            if grow_num_filters:
                num_filters += growth_rate

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net, num_filters

@slim.add_arg_scope
def _transition_block(inputs, num_filters, compression=1.0, last_one = False,
                      scope=None, outputs_collections=None):

    num_filters = int(num_filters * compression)
    with tf.variable_scope(scope, 'transition_blockx', [inputs]) as sc:
        net = inputs
        net = _conv(net, num_filters, 1, scope='blk')
        
        if last_one:
#            net = slim.avg_pool2d(net, 2, stride=1)
            net = slim.avg_pool2d(net, 2, stride=1, padding='SAME')
        else:
#            net = slim.avg_pool2d(net, 2)
            net = slim.avg_pool2d(net, 2, padding='SAME')

        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net, num_filters

def densenet(inputs,
             num_classes=None,
             reduction=None,
             growth_rate=None,
             num_filters=None,
             convs_each_num_layers=None,
             dropout_rate=None,
             data_format='NHWC',
             is_training=True,
             include_root_block=True,
             reuse=None,
             base_net=True,
             global_pool=True,
             scope=None):
    assert reduction is not None
    assert growth_rate is not None
    assert num_filters is not None
    assert convs_each_num_layers is not None

    compression = 1.0 - reduction
    num_dense_blocks = len(convs_each_num_layers)

    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    with tf.variable_scope(scope, 'densenetxxx', [inputs, num_classes],
                         reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                         is_training=is_training), \
             slim.arg_scope([slim.conv2d, _conv, _conv_block,
                         _dense_block, _transition_block], 
                         outputs_collections=end_points_collection), \
             slim.arg_scope([_conv], dropout_rate=dropout_rate):
             net = inputs
             
             if include_root_block:
             # initial convolution
#                 net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv1')
#                 net = slim.batch_norm(net)
#                 net = tf.nn.relu(net)
#                 net = slim.max_pool2d(net, 3, stride=2, padding='SAME')
                 net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                 net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
                 net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

             if base_net:
                 # blocks
                 for i in range(num_dense_blocks - 1):
                     # dense blocks
                     net, num_filters = _dense_block(net, convs_each_num_layers[i], num_filters,
                                            growth_rate,
                                            scope='dense_block' + str(i+1))
    
                     # Add transition_block
                     if i == num_dense_blocks - 2:
                         last_one = True
                     else:
                         last_one = False
                     net, num_filters = _transition_block(net, num_filters,
                                                 compression=compression, last_one=last_one,
                                                 scope='transition_block' + str(i+1))
    
                 net, num_filters = _dense_block(
                 net, convs_each_num_layers[-1], num_filters,
                 growth_rate,
                 scope='dense_block' + str(num_dense_blocks))
             else:
                 for i in range(num_dense_blocks - 1):
                     # dense blocks
                     net, num_filters = _dense_block(net, convs_each_num_layers[i], num_filters,
                                            growth_rate)
    
                     # Add transition_block
                     net, num_filters = _transition_block(net, num_filters,
                                                 compression=compression)
    
                 net, num_filters = _dense_block(
                 net, convs_each_num_layers[-1], num_filters, growth_rate)

             # final blocks
             if global_pool:
                 with tf.variable_scope('final_block', [inputs]):
                     net = slim.batch_norm(net)
                     net = tf.nn.relu(net)
                     net = _global_avg_pool2d(net, scope='global_avg_pool')
                     
             if num_classes is not None:
                 net = slim.conv2d(net, num_classes, 1,
                            biases_initializer=tf.zeros_initializer(),
                            scope='logits')

             end_points = slim.utils.convert_collection_to_dict(
                     end_points_collection)

             if num_classes is not None:
                 end_points['predictions'] = slim.softmax(net, scope='predictions')

             return net, end_points
densenet.default_image_size = 224

# =============================================================================
# def densenet121(inputs, data_format='NHWC', is_training=True, reuse=True):
#     return densenet(inputs,
#                   reduction=0.5,
#                   growth_rate=32,
#                   num_filters=64,
#                   convs_each_num_layers=[6,12,24,16],
#                   data_format=data_format,
#                   is_training=is_training,
#                   global_pool=False,
#                   reuse=reuse,
#                   scope='dense_121')
# densenet121.default_image_size = 224
# 
# 
# def densenet161(inputs, data_format='NHWC', is_training=True, reuse=None):
#     return densenet(inputs, 
#                   reduction=0.5,
#                   growth_rate=48,
#                   num_filters=96,
#                   convs_each_num_layers=[6,12,36,24],
#                   data_format=data_format,
#                   is_training=is_training,
#                   global_pool=False,
#                   reuse=reuse,
#                   scope='dense_161')
# densenet161.default_image_size = 224
# =============================================================================
   
class dense(Network):
    def __init__(self, batch_size=1, num_layers=121, reduction=0.5, dropout_rate=None):
        Network.__init__(self, batch_size=batch_size)   
        self._num_layers = num_layers
        self._dense_scope = 'densenet%d' % num_layers
#        if self._num_layers == 121:
#            self.num_filters = 64
#            self.growth_rate = 32
#        elif  self._num_layers == 161:
#            self.num_filters = 96
#            self.growth_rate = 48
        self.reduction = reduction
        self.dropout_rate = dropout_rate
    
    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be backpropagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [7, 7],
                                          name="crops")
        return crops
           
    def build_network(self, sess, is_training=True):
        if cfg.FLAGS.initializer == "truncated":
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)      
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
        
        if self._num_layers == 121:
            net_conv4, _ = densenet(self._image,
                              reduction=0.5,
                              growth_rate=32,
                              num_filters=64,
                              convs_each_num_layers=[6,12,24,16],
                              data_format='NHWC',
                              is_training=True,
                              global_pool=False,
                              reuse=None,
                              scope=self._dense_scope)
        elif self._num_layers == 169:
            net_conv4, _ = densenet(self._image, 
                              reduction=0.5,
                              growth_rate=48,
                              num_filters=96,
                              convs_each_num_layers=[6,12,36,24],
                              data_format='NHWC',
                              is_training=True,
                              global_pool=False,
                              reuse=None,
                              scope=self._dense_scope)
        else:
            # other numbers are not supported
            raise NotImplementedError

        self._act_summaries.append(net_conv4)
        self._layers['head'] = net_conv4
        with tf.variable_scope(self._dense_scope, self._dense_scope):
        
            # build the anchors for the image
            self._anchor_component()

            # rpn
            rpn = slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer
                            ,scope="rpn_conv/3x3")
            self._act_summaries.append(rpn)
            rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                      weights_initializer=initializer,
                                      padding='VALID', activation_fn=None
                                      , scope='rpn_cls_score')
            # change it so that the score has 2 as its channel size
            rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
            rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
            rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
            rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                      weights_initializer=initializer,
                                      padding='VALID', activation_fn=None
                                    , scope='rpn_bbox_pred')
            if is_training:
                rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
                rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
                # Try to have a determinestic order for the computing graph, for reproducibility
                with tf.control_dependencies([rpn_labels]):
                    rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
            else:
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")

        pool5 = self._crop_pool_layer(net_conv4, rois, "pool5")

        with slim.arg_scope(densenet_arg_scope(is_training=is_training)):
#            fc7, _ = resnet_v1.resnet_v1(pool5,
#                                       blocks[-1:],
#                                       global_pool=False,
#                                       include_root_block=False,
#                                       scope=self._dense_scope)
            if self._num_layers == 121:
                fc7, _ = densenet(pool5,
                                  reduction=0.5,
                                  growth_rate=32,
                                  num_filters=64,
                                  convs_each_num_layers=[16],
                                  is_training=is_training,
                                  global_pool=False,
                                  include_root_block=False,
                                  base_net=False,
                                  reuse=None
                                  ,scope=self._dense_scope)
            elif self._num_layers == 161:
                fc7, _ = densenet(pool5,
                                  reduction=0.5,
                                  growth_rate=48,
                                  num_filters=96,
                                  convs_each_num_layers=[24],
                                  is_training=is_training,
                                  include_root_block=False,
                                  global_pool=False,
                                  base_net=False,
                                  reuse=None)
#                                  ,scope=self._dense_scope)

        with tf.variable_scope(self._dense_scope, self._dense_scope):
            # Average pooling done by reduce_mean
            fc7 = tf.reduce_mean(fc7, axis=[1, 2])
            cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer,
                                           trainable=is_training, activation_fn=None
                                           , scope='cls_score')
            cls_prob = self._softmax_layer(cls_score, "cls_prob")
            bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                           trainable=is_training,
                                           activation_fn=None
                                            , scope='bbox_pred')
        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions["rois"] = rois
        self._score_summaries.update(self._predictions)
    
        return rois, cls_prob, bbox_pred

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._dense_scope + '/conv1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore
    
    def fix_variables(self, sess, pretrained_model):
        print('Fix dense layers..')
        with tf.variable_scope('Fix_DenseNet') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._dense_scope + "/conv1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._dense_scope + '/conv1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))
