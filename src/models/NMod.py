#!/usr/bin/env python3

# using http://cs231n.stanford.edu/reports/2017/pdfs/229.pdf
# and https://arxiv.org/pdf/1711.09017.pdf Gazenet

import tensorflow as tf
import util.gaze
import parameters

basefiltf = 16
basefilte = 32
class NMOD():
#eyes 224x60 # 3 2 2 -> 22x5
#face 224x224 # 2 2 2 #doing 3 first as well -> 22x22
#dilation_rate = (1,1)


#tf.layers.separable_conv2d


#TODO try initializer for separable conv2d
#TODO use canny edge detection? tf.image.sobel_edges(image) adds one dimension, last dimension is [0]dy[1]dx
#    G = np.hypot(Ix, Iy)
#    G = G / G.max() * 255      Intensity
#    theta = np.arctan2(Iy, Ix) Direction


    def conv3by3(self,tensor,name,filters,stride=1,reuse=None): #can be sobel filters

        conv3 = tf.layers.conv2d(tensor, filters, kernel_size=[1, 3], strides=stride, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_1',reuse=reuse)

        conv3 = tf.layers.conv2d(conv3, filters, kernel_size=[3, 1], strides=stride, padding='same',
                                 activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                 data_format="channels_first",name=name+'_2',reuse=reuse)

        return conv3

    def inception_module(self,tensor,name, filters,reuse=None, is_training=True):

        conv = tf.layers.conv2d(tensor, filters, kernel_size=[1,1], strides=[1,1], padding='same',
                            kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                            data_format='channels_first',name=name+'_inc1',reuse=reuse)

        conv3 = self.conv3by3(conv,name+'_conv3_', filters,reuse=reuse)

        conv5 = self.conv3by3(conv,name+'_conv5_1_', filters,reuse=reuse)
        conv5 = self.conv3by3(conv5,name+'_conv5_2_', filters,reuse=reuse)

        conv7 = self.conv3by3(conv,name+'_conv7_1_', filters,reuse=reuse)
        conv7 = self.conv3by3(conv7,name+'_conv7_2_', filters,reuse=reuse)
        conv7 = self.conv3by3(conv7,name+'_conv7_3_', filters, reuse=reuse)

        conv = tf.concat([conv, conv3, conv5, conv7],1)
        #conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)  # axis 1 because of channels first
        #conv = tf.nn.relu(conv)
        return conv

    def downsample(self, tensor,name, filters, kernel=[2,2], is_training=True):#level descend is valid conv 3x3x1 and max pool

        conv = tf.layers.conv2d(tensor, filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")


        conv2 = tf.layers.conv2d(conv, filters, kernel_size=kernel, strides=kernel, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")


        max = tf.layers.max_pooling2d(tensor, kernel, kernel,
                                      data_format="channels_first")
        cmax = tf.layers.conv2d(max, filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")

        conv = tf.concat([conv2,cmax],1)
        #conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)
        return conv

    def eyeconv(self, tensor,name,filters, reuse=None, kernel=[3,3], is_training=True):#filters start at 64
        #check with gaussian blur first aswell
        #sob = tf.image.sobel_edges(tensor)
        #sob = tf.math.sqrt(tf.reduce_sum(tf.math.pow(sob,2),4))
        #sobm = tf.reduce_max(sob,3,keepdims=True)
        #sob = sob/sobm
        #conv = self.inception_module(tensor,name+'_pre1',filters,reuse=reuse,is_training=is_training)#for preprocessing
        #ten = tf.concat([conv, sob, tensor], 1)

        #https://arxiv.org/pdf/1903.07296.pdf

        conv = tf.layers.conv2d(tensor, filters, kernel_size=[1, 1], strides=[1, 1], padding='same',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")



        #https://arxiv.org/pdf/1904.09459.pdf
        #60x90
        conv2 = tf.layers.conv2d(conv, 32, kernel_size=[5,5], strides=1, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first", name=name+'_conv2',reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2],
                                      data_format="channels_first")#43x23
        conv3 = tf.layers.conv2d(conv2, 32, kernel_size=[5,5], strides=1, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first", name=name+'_conv3',reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2],
                                      data_format="channels_first")#19x9
        conv4 = tf.layers.conv2d(conv3, 64, kernel_size=[5,5], strides=1, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first", name=name+'_conv4',reuse=reuse)#5x15
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        conv4 = tf.nn.relu(conv4)

        return conv4

    def __init__(self):#, learning_rate=parameters.LEARNING_RATE):
        self.f = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
        self.er = tf.placeholder(tf.float32, shape=(None, 3, 60, 224))
        self.le = tf.placeholder(tf.float32, shape=(None, 3, 60, 90))
        self.re = tf.placeholder(tf.float32, shape=(None, 3, 60, 90))
        self.h = tf.placeholder(tf.float32, shape=(None, 2))
        self.fl = tf.placeholder(tf.float32, shape=(None, 33, 2))
        self.g = tf.placeholder(tf.float32, shape=(None, 2))
        self.LR = tf.placeholder(tf.float32)
        learning_rate=self.LR

        self.drop = 0.5
        self.training = tf.placeholder(tf.bool)


        # However both of these are done by batch normalization to a certain degree + noisy dataset
        # Xavier init. uses mean 0 and variance of 1/Neurons (or Navg, being the average of input and output neurons, of a layer

        #self.base_init = tf.truncated_normal_initializer(stddev=0.001)  # Initialise weights 0.01 at max
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.001)  # Initialise regularisation Alexnet 0.01 // 0.001 seems to make more sense
        #trying with he initializer
        self.base_init = tf.contrib.layers.variance_scaling_initializer()

        #f = self.xception_f(self.f, 'face', is_training=self.training)
        #er = self.xception_er(self.er, 'eye-region', is_training=self.training)
        ile = self.eyeconv(self.le, 'ieye',32, is_training=self.training)
        ire = self.eyeconv(self.re, 'ieye',32, reuse=True, is_training=self.training)
        ele = self.eyeconv(self.le, 'eeye',32, is_training=self.training)
        ere = self.eyeconv(self.re, 'eeye',32, reuse=True, is_training=self.training)

        #er = self.eyeconv(self.le, 'eyer',32, is_training=self.training)
        #er = self.eyeconv(self.er, 'eye-region', 64, is_training=self.training)
        #f = self.eyeconv(self.f, 'face', 64, is_training=self.training)



        ire = tf.contrib.layers.flatten(ire)
        ile = tf.contrib.layers.flatten(ile)
        ire = tf.layers.dense(ire, units=256, activation=tf.nn.relu, name='RFC1')
        ile = tf.layers.dense(ile, units=256, activation=tf.nn.relu, name='LFC1')
        i = tf.concat([ire,ile],1)
        #i = tf.nn.relu(i)
        #i = tf.layers.dropout(i, rate=self.drop, training=self.training)

        #i = tf.layers.dense(i, units=256, name='F-FC2')
        i = tf.layers.dense(i, units=2, name='FC4')

        ere = tf.contrib.layers.flatten(ere)
        ele = tf.contrib.layers.flatten(ele)
        ere = tf.layers.dense(ere, units=256, activation=tf.nn.relu, name='eRFC1')
        ele = tf.layers.dense(ele, units=256, activation=tf.nn.relu, name='eLFC1')
        e = tf.concat([ere,ele],1)
        #e = tf.nn.relu(e)
        #e = tf.layers.dropout(e, rate=self.drop, training=self.training)

        #e = tf.layers.dense(e, units=256, name='eF-FC2')
        e = tf.layers.dense(e, units=2, name='eFC4')

        #head angle factor
        hf1 = tf.layers.dense(self.h, units=2, name='hf1')
        hf2 = tf.layers.dense(self.h, units=2, name='hf2')

        #BN both+1

        d = tf.atan(i*hf1-e*hf2)

        #angle factor
        af = tf.layers.dense(d, units=2, name='af')


        t = d+self.h*af#tf.concat()



        self.predictions = t#pred #tf.atan(pred)

        #pitch is y,yaw is x

        self.loss = tf.losses.mean_squared_error(self.g, self.predictions)#+tf.losses.get_regularization_loss()
        self.angular_loss = util.gaze.tensorflow_angular_error_from_pitchyaw(self.g, self.predictions)
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.extra_update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = self.trainer.minimize(self.loss)

