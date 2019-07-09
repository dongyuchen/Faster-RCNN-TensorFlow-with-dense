import time

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import lib.config.config as cfg
from lib.datasets import roidb as rdl_roidb
from lib.datasets.factory import get_imdb
from lib.datasets.imdb import imdb as imdb2
from lib.layer_utils.roi_data_layer import RoIDataLayer
from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1
from lib.nets.dense import dense
from lib.utils.timer import Timer

try:
  import cPickle as pickle
except ImportError:
  import pickle
import os

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    #返回一个用于训练的感兴趣区域的数据库
    if True:
        print('Appending horizontally-flipped training examples...')
        """其作用是将数据集中的每张图的所有bounding box标签进行水平翻转，
        然后将图片信息字典中的'flipped'置为True，并将这一新的字典添加进
        原始的roidb list中，这样图片信息列表的长度就变为了原来的2倍。
        最后将数据集实例中的_image_index成员（所有图片名的list）复制了一份，
        长度也变为了原来的2倍。"""
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb

def combined_roidb(imdb_names):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method("gt")
        print('Set proposal method: {:s}'.format("gt"))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = imdb2(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


class Train:
    def __init__(self):

        # Create network
#        if cfg.FLAGS.network == 'vgg16':
#            self.net = vgg16(batch_size=cfg.FLAGS.ims_per_batch)
#        if cfg.FLAGS.network == 'resnetv1':
#            self.net = resnetv1(batch_size=cfg.FLAGS.ims_per_batch,num_layers=101)
        if cfg.FLAGS.network == 'dense':
            self.net = dense(batch_size=cfg.FLAGS.ims_per_batch,num_layers=121)
        else:
            raise NotImplementedError

        self.imdb, self.roidb = combined_roidb("voc_2007_trainval")

        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        self.output_dir = cfg.get_output_dir(self.imdb, 'default')


    def train(self):

        # Create session
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
#        sess = tf.Session(config=tfconfig)
        sess = tf.Session()

        with sess.graph.as_default():

            tf.set_random_seed(cfg.FLAGS.rng_seed)
            layers = self.net.create_architecture(sess, "TRAIN", self.imdb.num_classes, tag='default')
            loss = layers['total_loss']
            lr = tf.Variable(cfg.FLAGS.learning_rate, trainable=False)
            momentum = cfg.FLAGS.momentum
            optimizer = tf.train.MomentumOptimizer(lr, momentum)

            gvs = optimizer.compute_gradients(loss)

            # Double bias
            # Double the gradient of the bias if set
            if cfg.FLAGS.double_bias:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult'):
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.FLAGS.double_bias and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = optimizer.apply_gradients(final_gvs)
            else:
                train_op = optimizer.apply_gradients(gvs)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
#             writer = tf.summary.FileWriter(cfg.FLAGS.tbdir, sess.graph)
#             valwriter = tf.summary.FileWriter(self.tbvaldir)

        # Load weights
        # Fresh train directly from ImageNet weights
        print('Loading initial model weights from {:s}'.format(cfg.FLAGS.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(cfg.FLAGS.pretrained_model)
        # Get the variables to restore, ignorizing the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables, var_keep_dic)

        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, cfg.FLAGS.pretrained_model)
        print('Loaded.')
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights
        self.net.fix_variables(sess, cfg.FLAGS.pretrained_model)
        print('Fixed.')
        sess.run(tf.assign(lr, cfg.FLAGS.learning_rate))
        last_snapshot_iter = 0

        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        while iter < cfg.FLAGS.max_iters + 1:
#            print("iter:",iter)
            # Learning rate
            if iter == cfg.FLAGS.step_size + 1:
                # Add snapshot here before reducing the learning rate
                # self.snapshot(sess, iter)
                sess.run(tf.assign(lr, cfg.FLAGS.learning_rate * cfg.FLAGS.gamma))

            timer.tic()
            # Get training data, one batch at a time
            blobs = self.data_layer.forward()

            # Compute the graph without summary
            rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = self.net.train_step(sess, blobs, train_op)
#            print("rpn_loss_cls:",rpn_loss_cls)
            timer.toc()
            iter += 1
#            print("iter加一后：",iter)

            # Display training information
            if iter % (cfg.FLAGS.display) == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n ' % \
                      (iter, cfg.FLAGS.max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if iter % cfg.FLAGS.snapshot_iterations == 0:
                self.snapshot(sess, iter )

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'res_faster_rcnn_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = 'res_faster_rcnn_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.data_layer._cur
        # current shuffled indeces of the database
        perm = self.data_layer._perm

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename


if __name__ == '__main__':
    train = Train()
    train.train()
