#!/usr/bin/env python3

import tensorflow as tf
import util.gaze
import parameters

basefilt = 16


class InceptionV4():
    """An example neural network architecture combining Inception_v4,
    https://arxiv.org/pdf/1602.07261.pdf.

    ******************************************************************
    ************************InceptionV4_v_2***************************
    *with iteration times for inception_a, _b, _c modules of 2, 4, 2;*
    *kernel_initializer and kernel_regularizer of both 0.001;        *
    *activation functions are relu and softmax;                      *
    *batch_normalization added afer each dense layer                 *
    *face, head pose all matters but more weight for head pose       *
    *learning_rate=0.0001 (default for adam optimizer is 0.001)      *
    ******************************************************************
    """

    def conv3by3(self, tensor, name, filters, strides=[1, 1], padding='same', reuse=None, is_training=True, bn=False):
        with tf.variable_scope(name + '1'):
            conv3 = tf.layers.conv2d(tensor, filters=filters, kernel_size=[1, 3], strides=strides, padding=padding,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                     data_format="channels_first", reuse=reuse)
        if bn:
            conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        else:
            pass

        with tf.variable_scope(name + '2'):
            conv3 = tf.layers.conv2d(conv3, filters=filters, kernel_size=[3, 1], strides=strides, padding=padding,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                     data_format="channels_first", reuse=reuse)

        if bn:
            conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        else:
            pass

        return conv3

    def conv7by7(self, tensor, name, filters, strides=[1, 1], padding='same', reuse=None, is_training=True, bn=False):
        with tf.variable_scope(name + '1'):
            conv7 = tf.layers.conv2d(tensor, filters=filters, kernel_size=[1, 7], strides=strides, padding=padding,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                     data_format="channels_first", reuse=reuse)

        if bn:
            conv7 = tf.layers.batch_normalization(conv7, training=is_training, axis=1)
        else:
            pass

        with tf.variable_scope(name + '2'):
            conv7 = tf.layers.conv2d(conv7, filters=filters, kernel_size=[7, 1], strides=strides, padding=padding,
                                     activation=tf.nn.relu,
                                     kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                     data_format="channels_first", reuse=reuse)

        if bn:
            conv7 = tf.layers.batch_normalization(conv7, training=is_training, axis=1)
        else:
            pass

        conv7 = tf.nn.relu(conv7)
        return conv7

    def stem_module_large(self, tensor, name, is_training=True, reuse=None):
        """
        The input module stem of the Inception_v4 model.
        Input: 224*224*3, output:25*25*384
        Input left-eye or right-eye: 90*60*3, output: 8*5*384
        """
        conv = tf.layers.conv2d(tensor, 32, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                activation=tf.nn.relu,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                data_format='channels_first',
                                name=name + '1conv_1',
                                reuse=reuse)
        # conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)

        conv = self.conv3by3(conv, name=name + '1conv_2', filters=32, padding='valid',
                             reuse=reuse, is_training=is_training)  # eye output: 42, 27, 32; face output: 109, 109, 32

        conv = self.conv3by3(conv, name=name + '1conv_3', filters=64, padding='same',
                             reuse=reuse, is_training=is_training)  # eye output: 42, 27, 64; face output: 109, 109, 64

        maxpool1 = tf.layers.max_pooling2d(conv, [3, 3], [2, 2], data_format='channels_first', padding='valid')
        # eye output = 20, 13, 64; face output: 54, 54, 64
        conv1 = tf.layers.conv2d(conv, 92, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '1conv1_1',
                                 reuse=reuse)
        # eye output = 20, 13, 96 ; face output: 54, 54, 96
        # conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)

        conv = tf.concat([maxpool1, conv1], 1)
        # eye output = 20, 13, 160; face output: 54, 54, 160

        conv2 = tf.layers.conv2d(conv, 64, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '1conv2_1',
                                 reuse=reuse)  # eye output = 20, 12, 64; face output: 54, 54, 64
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)

        conv2 = self.conv3by3(conv2, name=name + '1conv2_2', filters=96, padding='valid',
                              reuse=reuse, is_training=is_training)
        # eye output = 18, 10, 96; face output: 52, 52, 96

        conv3 = tf.layers.conv2d(conv, 64, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '1conv3_1',
                                 reuse=reuse)  # eye output = 20, 12, 64; face output: 54, 54, 64
        # conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)

        conv3 = self.conv7by7(conv3, name=name + '1conv3_2', filters=64, padding='same',
                              reuse=reuse, is_training=is_training)
        # eye output = 20, 12, 64; face output: 54, 54, 64
        conv3 = self.conv3by3(conv3, name=name + '1conv3_3', filters=96, padding='valid',
                              reuse=reuse, is_training=is_training)
        # eye output = 18, 10, 96; face output: 52, 52, 96

        conv = tf.concat([conv2, conv3], 1)  # eye output = 18, 10, 192; face output: 52, 52, 192

        conv4 = tf.layers.conv2d(conv, 192, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '1conv4_1',
                                 reuse=reuse)
        # eye output = 8, 4, 192; face output: 25, 25, 192
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)

        maxpool2 = tf.layers.max_pooling2d(conv, [3, 3], 2, data_format='channels_first')
        # eye output = 8, 4, 192; face output: 25, 25, 192

        conv = tf.concat([conv4, maxpool2], 1)  # eye output = 8, 4, 384; face output: 25, 25, 384

        return conv

    def stem_module_small(self, tensor, name, is_training=True, reuse=None):
        """
        The input module stem of the Inception_v4 model for eye pic.
        """
        conv = tf.layers.conv2d(tensor, 32, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                data_format='channels_first',
                                name=name + '11conv_1',
                                reuse=reuse)
        # eye output 90, 60, 32
        # conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)
        conv = self.conv3by3(conv, name=name + '11conv_2', filters=32, padding='valid',
                             reuse=reuse, is_training=is_training)  # eye output 88, 58, 32
        conv = self.conv3by3(conv, name=name + '11conv_3', filters=64, padding='same',
                             reuse=reuse, is_training=is_training)  # eye output: 88, 58, 64

        # maxpool1 = tf.layers.max_pooling2d(conv, [3, 3], [2, 2], data_format='channels_first',
        #                                    padding='valid')
        # conv1_1 = tf.layers.conv2d(conv, 80, kernel_size=[3, 3], strides=[2, 2], padding='valid',
        #                           kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
        #                           data_format='channels_first', name=name+'1conv1_1', reuse=reuse)
        conv1_1 = self.conv3by3(conv, name=name + '11conv1_1', filters=80,
                                reuse=reuse, is_training=is_training, bn=True)
        # eye output 88, 58, 80
        conv1_2 = self.conv3by3(conv, name=name + '11conv1_2', filters=80,
                                reuse=reuse, is_training=is_training, bn=True)
        # eye output 88, 58, 80

        conv = tf.concat([conv1_1, conv1_2], 1)
        # eye output 88, 58, 160

        conv2 = tf.layers.conv2d(conv, 64, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '11conv2_1',
                                 reuse=reuse)
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)

        conv2 = self.conv3by3(conv2, name=name + '11conv2_2', filters=96, padding='valid',
                              reuse=reuse, is_training=is_training)
        # eye output 86, 54, 96

        conv3 = tf.layers.conv2d(conv, 64, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '11conv3_1',
                                 reuse=reuse)  # eye output = 20, 12, 64; face output: 54, 54, 64
        # conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)

        conv3 = self.conv7by7(conv3, name=name + '11conv3_2', filters=64, padding='same',
                              reuse=reuse, is_training=is_training)
        conv3 = self.conv3by3(conv3, name=name + '11conv3_3', filters=96, padding='valid',
                              reuse=reuse, is_training=is_training)
        # eye output 86, 54, 96

        conv = tf.concat([conv2, conv3], 1)  # eye output = 86, 54, 192

        conv4 = tf.layers.conv2d(conv, 192, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '11conv4_1',
                                 reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        maxpool2 = tf.layers.max_pooling2d(conv, [3, 3], 2, data_format='channels_first')
        # eye output 42, 26, 192

        conv = tf.concat([conv4, maxpool2], 1)
        # eye output = 42, 26, 384

        return conv

    def stem_f(self, tensor, name, is_training=True, reuse=None):
        """
        The input module stem of the Inception_Res_v1 model.
        Input: 224*224*3, output:25*25*384
        """
        conv = tf.layers.conv2d(tensor, 32, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                activation=tf.nn.relu,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                data_format='channels_first',
                                name=name + '1conv_1',
                                reuse=reuse)  # 111, 111, 32
        conv = tf.layers.conv2d(conv, 32, kernel_size=[3, 3], strides=[1, 1], padding='valid',
                                activation=tf.nn.relu,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                data_format='channels_first',
                                name=name + '1conv_2',
                                reuse=reuse)  # 109, 109, 32
        conv = tf.layers.conv2d(conv, 64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=self.base_init,
                                kernel_regularizer=self.reg_init,
                                data_format='channels_first',
                                name=name + '1conv_3',
                                reuse=reuse)  # 109, 109, 64
        return conv


    def inception_a_module(self, tensor, name, is_training=True, reuse=None):
        """Modules of the pure Inception_v4 network. This is the Inception-A block.
        :param tensor: input tensor from the output of stem_module, 42, 26, 384
        :param reuse: reuse or not, example: tf.layers.conv2d(reuse=None)
        :return: eye output = 42, 26, 384; ; face output: 25, 25, 384
        """
        avgpool = tf.contrib.layers.avg_pool2d(tensor, kernel_size=[2, 2], stride=1, padding='SAME',
                                               data_format='NCHW')

        conv1 = tf.layers.conv2d(avgpool, 96, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '2conv1',
                                 reuse=reuse)

        conv2 = tf.layers.conv2d(tensor, 96, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '2conv2',
                                 reuse=reuse)
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)

        conv3 = tf.layers.conv2d(tensor, 64, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '2conv3_1',
                                 reuse=reuse)
        # conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)

        conv3 = self.conv3by3(conv3, name=name + '2conv3_2', filters=96,
                              reuse=reuse, is_training=is_training, bn=True)

        conv4 = tf.layers.conv2d(tensor, 64, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '2conv4_1',
                                 reuse=reuse)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        conv4 = self.conv3by3(conv4, name=name + '2conv4_2', filters=96,
                              reuse=reuse, is_training=is_training)
        conv4 = self.conv3by3(conv4, name=name + '2conv4_3', filters=96,
                              reuse=reuse, is_training=is_training, bn=True)

        conv = tf.concat([conv1, conv2, conv3, conv4], 1)

        return conv  # eye output 42, 26, 384; face output 25, 25, 384

    def reduction_a_module(self, tensor, name, k, l, m, n, is_training=True, reuse=None):
        """
        Reduction module for Inception_v4 network.
        :param tensor: input tensor from the output of inception_a_module
        :param k: number of filters for conv2_1
        :param l: conv2_2
        :param m: conv2_3
        :param n: conv1
        :param is_training:
        :return: eye output = 20, 12, 1024; face output: 12, 12, 1024
        """
        maxpool = tf.layers.max_pooling2d(tensor, [3, 3], 2, data_format='channels_first')

        conv1 = tf.layers.conv2d(tensor, n, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '3conv1',
                                 reuse=reuse)
        # conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)

        conv2 = tf.layers.conv2d(tensor, filters=k, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '3conv2_1',
                                 reuse=reuse)
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)

        conv2 = self.conv3by3(conv2, name=name + '3conv2_2', filters=l,
                              reuse=reuse, is_training=is_training)
        conv2 = tf.layers.conv2d(conv2, m, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '3conv2_3',
                                 reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)

        conv = tf.concat([maxpool, conv1, conv2], 1)

        return conv  # eye output  20, 12, 1024; face output 12, 12, 1024

    def inception_b_module(self, tensor, name, is_training=True, reuse=None):
        """
        Modules of the pure Inception-v4 network. This is the Inception-B bloc
        """
        avgpool = tf.contrib.layers.avg_pool2d(tensor, kernel_size=[2, 2], stride=1, padding='SAME',
                                               data_format='NCHW')
        conv1 = tf.layers.conv2d(avgpool, filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '4conv1',
                                 reuse=reuse)
        # conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)

        conv2 = tf.layers.conv2d(tensor, filters=384, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '4conv2',
                                 reuse=reuse)
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)

        conv3 = tf.layers.conv2d(tensor, filters=192, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '4conv3_1',
                                 reuse=reuse)
        # conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv3 = tf.layers.conv2d(conv3, filters=224, kernel_size=[1, 7], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format="channels_first",
                                 name=name + '4conv3_2',
                                 reuse=reuse)
        # conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv3 = tf.layers.conv2d(conv3, filters=256, kernel_size=[1, 7], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format="channels_first",
                                 name=name + '4conv3_3',
                                 reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)

        conv4 = tf.layers.conv2d(tensor, filters=192, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '4conv4_1',
                                 reuse=reuse)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        conv4 = tf.layers.conv2d(conv4, filters=192, kernel_size=[1, 7], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format="channels_first",
                                 name=name + '4conv4_2',
                                 reuse=reuse)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        conv4 = tf.layers.conv2d(conv4, filters=224, kernel_size=[7, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format="channels_first",
                                 name=name + '4conv4_3',
                                 reuse=reuse)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        conv4 = tf.layers.conv2d(conv4, filters=224, kernel_size=[1, 7], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format="channels_first",
                                 name=name + '4conv4_4',
                                 reuse=reuse)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        conv4 = tf.layers.conv2d(conv4, filters=256, kernel_size=[7, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format="channels_first",
                                 name=name + '4conv4_5',
                                 reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)

        conv = tf.concat([conv1, conv2, conv3, conv4], 1)

        return conv

    def reduction_b_module(self, tensor, name, is_training=True, reuse=None):
        """the reduction module used by the pure Inception-v4 network

        :param tensor: input from the output of inception_b_module, only for face: 12, 12, 1024
        :param is_training:
        :param reuse:
        :return: eye output = 9, 5, 1536; face output: 5, 5, 1536
        """
        maxpool = tf.layers.max_pooling2d(tensor, [3, 3], 2, data_format='channels_first')  # face: 5, 5, 1024

        conv1 = tf.layers.conv2d(tensor, filters=192, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '5conv1_1',
                                 reuse=reuse)
        # conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)
        conv1 = tf.layers.conv2d(conv1, 192, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '5conv1_2',
                                 reuse=reuse)
        # conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)

        conv2 = tf.layers.conv2d(tensor, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '5conv2_1',
                                 reuse=reuse)
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv2 = self.conv7by7(conv2, name=name + '5conv2_2', filters=320,
                              reuse=reuse, is_training=is_training)
        conv2 = tf.layers.conv2d(conv2, 320, kernel_size=[3, 3], strides=[2, 2], padding='valid',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '5conv2_3',
                                 reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)

        conv = tf.concat([maxpool, conv1, conv2], 1)  # 5, 5, 1536

        return conv

    def inception_c_module(self, tensor, name, is_training=True, reuse=None):
        """Modules of the pure Inceptionv4 network. This is the Inception-C block.

        :param tensor: face from reduction_b_module 5, 5, 1536
        """
        avgpool = tf.contrib.layers.avg_pool2d(tensor, kernel_size=[2, 2], stride=1, padding='SAME',
                                               data_format='NCHW')
        conv1 = tf.layers.conv2d(avgpool, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '6conv1',
                                 reuse=reuse)
        # conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)

        conv2 = tf.layers.conv2d(tensor, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '6conv2',
                                 reuse=reuse)
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)

        conv3 = tf.layers.conv2d(tensor, filters=384, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '6conv3',
                                 reuse=reuse)
        # conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv3_1 = tf.layers.conv2d(conv3, filters=256, kernel_size=[1, 3], strides=[1, 1], padding='same',
                                   activation=tf.nn.relu,
                                   kernel_initializer=self.base_init,
                                   kernel_regularizer=self.reg_init,
                                   data_format="channels_first",
                                   name=name + '6conv3_1',
                                   reuse=reuse)
        conv3_1 = tf.layers.batch_normalization(conv3_1, training=is_training, axis=1)
        conv3_2 = tf.layers.conv2d(conv3, filters=256, kernel_size=[3, 1], strides=[1, 1], padding='same',
                                   activation=tf.nn.relu,
                                   kernel_initializer=self.base_init,
                                   kernel_regularizer=self.reg_init,
                                   data_format="channels_first",
                                   name=name + '6conv3_2',
                                   reuse=reuse)
        conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_training, axis=1)

        conv4 = tf.layers.conv2d(tensor, filters=384, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=self.base_init,
                                 kernel_regularizer=self.reg_init,
                                 data_format='channels_first',
                                 name=name + '6conv4',
                                 reuse=reuse)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        conv4 = self.conv3by3(conv4, name=name + '6conv4_', filters=512,
                              reuse=reuse, is_training=is_training)
        conv4_1 = tf.layers.conv2d(conv4, filters=256, kernel_size=[3, 1], strides=[1, 1], padding='same',
                                   activation=tf.nn.relu,
                                   kernel_initializer=self.base_init,
                                   kernel_regularizer=self.reg_init,
                                   data_format="channels_first",
                                   name=name + '6conv4_1',
                                   reuse=reuse)
        conv4_1 = tf.layers.batch_normalization(conv4_1, training=is_training, axis=1)
        conv4_2 = tf.layers.conv2d(conv4, filters=256, kernel_size=[1, 3], strides=[1, 1], padding='same',
                                   activation=tf.nn.relu,
                                   kernel_initializer=self.base_init,
                                   kernel_regularizer=self.reg_init,
                                   data_format="channels_first",
                                   name=name + '6conv4_2',
                                   reuse=reuse)
        conv4_2 = tf.layers.batch_normalization(conv4_2, training=is_training, axis=1)

        conv = tf.concat([conv1, conv2, conv3_1, conv3_2, conv4_1, conv4_2], 1)

        return conv  # eye output = 9, 5, 1536; face output: 5, 5, 1536

    def __init__(self, learning_rate=parameters.LEARNING_RATE):
        self.f = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
        self.er = tf.placeholder(tf.float32, shape=(None, 3, 224, 60))
        self.le = tf.placeholder(tf.float32, shape=(None, 3, 90, 60))
        self.re = tf.placeholder(tf.float32, shape=(None, 3, 90, 60))
        self.h = tf.placeholder(tf.float32, shape=(None, 2))
        self.fl = tf.placeholder(tf.float32, shape=(None, 33, 2))
        self.g = tf.placeholder(tf.float32, shape=(None, 2))

        self.training = tf.placeholder(tf.bool)

        self.base_init = tf.truncated_normal_initializer(stddev=0.001)  # Initialise weights 0.01 at max
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.001)  # Initialise regularisation Alexnet 0.01
        # // 0.001 seems to make more sense

        # fa = self.stem_module_large(self.f, name='FA_INPUT', is_training=self.training)
        le = self.stem_module_small(self.le, name='E_INPUT', is_training=self.training)
        re = self.stem_module_small(self.re, name='E_INPUT', reuse=True, is_training=self.training)

        # fa = self.inception_a_module(fa, name='FAa')
        for idx in range(2):
            # fa = self.inception_a_module(fa, name='FAa' + str(idx), is_training=self.training)
            le = self.inception_a_module(le, name='Ea' + str(idx), is_training=self.training)
            re = self.inception_a_module(re, name='Ea' + str(idx), reuse=True, is_training=self.training)

        # fa = self.reduction_a_module(fa, 'FA_RE', 192, 224, 256, 384, is_training=self.training)
        le = self.reduction_a_module(le, 'LE_RE', 192, 224, 256, 384, is_training=self.training)
        re = self.reduction_a_module(re, 'LE_RE', 192, 224, 256, 384, reuse=True, is_training=self.training)

        # fa = self.inception_b_module(fa, name='FAb')
        for idx in range(3):
            # fa = self.inception_b_module(fa, name='FAb' + str(idx), is_training=self.training)
            le = self.inception_b_module(le, name='Eb' + str(idx), is_training=self.training)
            re = self.inception_b_module(re, name='Eb' + str(idx), reuse=True, is_training=self.training)

        # fa = self.reduction_b_module(fa, name='FA_RE2', is_training=self.training)
        le = self.reduction_b_module(le, name='LE_RE2', is_training=self.training)
        re = self.reduction_b_module(re, name='LE_RE2', reuse=True, is_training=self.training)

        # fa = self.inception_c_module(fa, name='FAc')
        for idx in range(2):
            # fa = self.inception_c_module(fa, name='FAc' + str(idx), is_training=self.training)
            le = self.inception_c_module(le, name='Ec' + str(idx), is_training=self.training)
            re = self.inception_c_module(re, name='Ec' + str(idx), reuse=True, is_training=self.training)

        # fa = tf.contrib.layers.avg_pool2d(fa, kernel_size=[5, 5], data_format='NCHW')
        le = tf.contrib.layers.avg_pool2d(le, kernel_size=[9, 5], data_format='NCHW')
        re = tf.contrib.layers.avg_pool2d(re, kernel_size=[9, 5], data_format='NCHW')

        # fa = tf.nn.dropout(fa, keep_prob=0.8)
        le = tf.nn.dropout(le, keep_prob=0.8)
        re = tf.nn.dropout(re, keep_prob=0.8)

        # fa = tf.contrib.layers.flatten(fa)
        # fl = tf.contrib.layers.flatten(self.fl)
        le = tf.contrib.layers.flatten(le)
        re = tf.contrib.layers.flatten(re)

        # fa = tf.layers.dense(fa, units=512, activation=tf.nn.softmax, name='DC-FA')
        # fa = tf.layers.batch_normalization(fa, training=self.training, axis=1)
        le = tf.layers.dense(le, units=1024, activation=tf.nn.softmax, name='DC-LE')
        le = tf.layers.batch_normalization(le, training=self.training, axis=1)
        re = tf.layers.dense(re, units=1024, activation=tf.nn.softmax, name='DC-LE', reuse=True) #TODO:formaer is softmax now is relu
        re = tf.layers.batch_normalization(re, training=self.training, axis=1)

        x = tf.concat([le, re], 1)
        x = tf.layers.dense(x, units=512, activation=None, name='FC-E1')
        x = tf.layers.batch_normalization(x, training=self.training, axis=1)
        x =tf.nn.relu(x)

        # x = tf.layers.dense(x, units=256, activation=tf.nn.relu, name='FC-E2')
        # x = tf.layers.dense(x, units=128, activation=tf.nn.relu, name='FC-E3')

        # fa = tf.layers.dense(fa, units=128, activation=tf.nn.relu, name='FC-FA1')
        # fa = tf.layers.batch_normalization(fa, training=self.training, axis=1)
        # fa = tf.layers.dense(fa, units=64, activation=tf.nn.relu, name='FC-FA2')

        # fl = tf.layers.dense(fl, units=33, activation=tf.nn.relu, name='FC-FL1')
        # fl = tf.layers.dense(fl, units=15, activation=tf.nn.relu, name='FC-FL2')

        # x = tf.concat([x, fa], 1)
        x = tf.layers.dense(x, units=256, activation=None, name='FC1')
        x = tf.layers.batch_normalization(x, training=self.training, axis=1)
        x = tf.nn.relu(x)
        # x = tf.layers.dense(x, units=128, activation=tf.nn.relu, name='FC2')
        # x = tf.layers.batch_normalization(x, training=self.training, axis=1)
        x = tf.layers.dense(x, units=64, activation=None, name='FC3')
        x = tf.layers.batch_normalization(x, training=self.training, axis=1)
        x = tf.nn.relu(x)

        # h = tf.tan(self.h)
        # increased the neurons for head pose
        # h = tf.layers.dense(h, units=4, activation=tf.nn.relu)
        # h = tf.layers.batch_normalization(h, training=self.training, axis=1)
        # h = tf.layers.dense(h, units=8, activation=tf.nn.relu)
        # h = tf.layers.batch_normalization(h, training=self.training, axis=1)
        # x = tf.concat([x, h], 1)

        x = tf.layers.dense(x, units=2, name='FC4')
        x = tf.layers.batch_normalization(x, training=self.training, axis=1)

        self.predictions = tf.atan(x)

        # self.correct_prediction = tf.equal(tf.argmax(tf.atan(x), 1), tf.argmax(self.g, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.loss = tf.losses.mean_squared_error(self.g, self.predictions)
        self.angular_loss = util.gaze.tensorflow_angular_error_from_pitchyaw(self.g, self.predictions)
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = self.trainer.minimize(self.loss)