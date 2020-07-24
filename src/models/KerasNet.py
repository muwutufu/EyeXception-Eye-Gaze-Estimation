#!/usr/bin/env python3

# using http://cs231n.stanford.edu/reports/2017/pdfs/229.pdf
# and https://arxiv.org/pdf/1711.09017.pdf Gazenet

import tensorflow as tf
import util.gaze
import parameters

basefiltf = 16
basefilte = 32
class KNet():
#eyes 224x60 # 3 2 2 -> 22x5
#face 224x224 # 2 2 2 #doing 3 first as well -> 22x22
#dilation_rate = (1,1)


#tf.layers.separable_conv2d


#TODO try initializer for separable conv2d


    def __init__(self, drop=0.3):#, learning_rate=parameters.LEARNING_RATE):
        self.f = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
        self.er = tf.placeholder(tf.float32, shape=(None, 3, 224, 60))
        self.le = tf.placeholder(tf.float32, shape=(None, 3, 90, 60))
        self.re = tf.placeholder(tf.float32, shape=(None, 3, 90, 60))
        self.h = tf.placeholder(tf.float32, shape=(None, 2))
        self.fl = tf.placeholder(tf.float32, shape=(None, 33, 2))
        self.g = tf.placeholder(tf.float32, shape=(None, 2))
        self.drop = drop
        self.training = tf.placeholder(tf.bool)
        self.LR = tf.placeholder(tf.float32)
        learning_rate=self.LR

        # However both of these are done by batch normalization to a certain degree + noisy dataset
        # Xavier init. uses mean 0 and variance of 1/Neurons (or Navg, being the average of input and output neurons, of a layer

        self.base_init = tf.truncated_normal_initializer(stddev=0.001)  # Initialise weights 0.01 at max
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.00001)  # Initialise regularisation Alexnet 0.01 // 0.001 seems to make more sense
        #trying with he initializer
        #self.base_init = tf.contrib.layers.variance_scaling_initializer()



        #TODO: reuse variables for eyes
        #with tf.variable_scope("foo") as scope:
        #    v = tf.get_variable("v", [1])
        #    scope.reuse_variables()
        #    v1 = tf.get_variable("v", [1])
        #assert v1 == v
        #FDnet = tf.keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling='avg')#, classes=1000)
        #ERDnet = tf.keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None,
        #                                                   input_shape=(3,224,60), pooling='avg')

        #REDnet = tf.keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None,
        #                                                   input_shape=None, pooling='avg')
        #f = self.xception_f(self.f, 'face', is_training=self.training)


        le = tf.layers.flatten(self.le,data_format='channels_first')
        pred = tf.layers.dense(le, units=2, name='FC4')#LEDnet.predict(self.le, batch_size=parameters.BATCH_SIZE, steps=1)


        self.predictions = pred#pred #

        #pitch is y,yaw is x

        self.loss = tf.losses.mean_squared_error(self.g, self.predictions)+tf.losses.get_regularization_loss()
        self.angular_loss = util.gaze.tensorflow_angular_error_from_pitchyaw(self.g, self.predictions)
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.extra_update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = self.trainer.minimize(self.loss)

        #TODO try with dropout!