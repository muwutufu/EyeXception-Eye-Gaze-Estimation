#!/usr/bin/env python3

# using http://cs231n.stanford.edu/reports/2017/pdfs/229.pdf
# and https://arxiv.org/pdf/1711.09017.pdf Gazenet

import tensorflow as tf
import util.gaze
import parameters
import math as m

basefiltf = parameters.BASEFILTER_SIZE
basefilte = parameters.BASEFILTER_SIZE
class GazeNetPlus():
#eyes 224x60 # 3 2 2 -> 22x5
#face 224x224 # 2 2 2 #doing 3 first as well -> 22x22
#dilation_rate = (1,1)



    def xception_f(self, tensor, name, reuse=None, is_training=True):

        #BEGIN ENTRY FLOW
        print('in output shape', tensor.get_shape())
        conv = tf.layers.conv2d(tensor, basefiltf, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv1',reuse=reuse)

        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)
        print('conv output shape', conv.get_shape())

        conv2 = tf.layers.conv2d(conv, basefiltf*2, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv2',reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        print('conv2 output shape', conv2.get_shape())
        conv3 = tf.layers.separable_conv2d(conv2, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv3',reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        print('conv3 output shape', conv3.get_shape())
        conv4 = tf.layers.separable_conv2d(conv3, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv4',reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)
        print('conv4 output shape', conv4.get_shape())
        max1 = tf.layers.max_pooling2d(conv4, pool_size=3,strides=2, padding='same',
                              data_format="channels_first")
        print('max1 output shape', max1.get_shape())
        convs1 = tf.layers.conv2d(conv2, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv1',reuse=reuse)
        convs1 = tf.layers.batch_normalization(convs1, training=is_training, axis=1) #applying BN before relu here and afterwards since it's still a topic of debate, I will assume minimal impact
        print('convs1 output shape', convs1.get_shape())
        l1 = tf.concat([max1,convs1],1)

        l1_b = tf.nn.relu(l1)

        conv5 = tf.layers.separable_conv2d(l1_b, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv5',reuse=reuse)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training, axis=1)
        conv6 = tf.layers.separable_conv2d(conv5, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv6',reuse=reuse)
        conv6 = tf.layers.batch_normalization(conv6, training=is_training, axis=1)
        max2 = tf.layers.max_pooling2d(conv6, pool_size=3, strides=2,padding='same',
                              data_format="channels_first")

        convs2 = tf.layers.conv2d(l1, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv2',reuse=reuse)
        convs2 = tf.layers.batch_normalization(convs2, training=is_training, axis=1)
        l2 = tf.concat([max2, convs2],1)

        l2_b = tf.nn.relu(l2)

        conv7 = tf.layers.separable_conv2d(l2_b, basefiltf*12, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv7',reuse=reuse)
        conv7 = tf.layers.batch_normalization(conv7, training=is_training, axis=1)
        conv8 = tf.layers.separable_conv2d(conv7, basefiltf*12, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv8',reuse=reuse)
        conv8 = tf.layers.batch_normalization(conv8, training=is_training, axis=1)
        max3 = tf.layers.max_pooling2d(conv8, pool_size=4, strides=2,padding='same',
                              data_format="channels_first")

        convs3 = tf.layers.conv2d(l2, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv3',reuse=reuse)
        convs3 = tf.layers.batch_normalization(convs3, training=is_training, axis=1)

        l3 = tf.concat([max3, convs3],1)

        #END ENTRY FLOW

        #BEGIN MIDDLE FLOW

        M1 = self.Middle_flow(l3, name+'_M1', reuse=reuse, is_training=is_training)
        M2 = self.Middle_flow(M1, name + '_M2', reuse=reuse, is_training=is_training)
        M3 = self.Middle_flow(M2, name + '_M3', reuse=reuse, is_training=is_training)
        M4 = self.Middle_flow(M3, name + '_M4', reuse=reuse, is_training=is_training)
        M5 = self.Middle_flow(M4, name + '_M5', reuse=reuse, is_training=is_training)
        M6 = self.Middle_flow(M5, name + '_M6', reuse=reuse, is_training=is_training)
        M7 = self.Middle_flow(M6, name + '_M7', reuse=reuse, is_training=is_training)
        M8 = self.Middle_flow(M7, name + '_M8', reuse=reuse, is_training=is_training)

        #END MIDDLE FLOW

        #BEGIN END FLOW

        l4 = tf.nn.relu(M8)

        convend1 = tf.layers.separable_conv2d(l4, basefiltf*12, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_convend5',reuse=reuse)
        convend1 = tf.layers.batch_normalization(convend1, training=is_training, axis=1)
        convend2 = tf.layers.separable_conv2d(convend1, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_convend6',reuse=reuse)
        convend2 = tf.layers.batch_normalization(convend2, training=is_training, axis=1)
        max3 = tf.layers.max_pooling2d(convend2, pool_size=3, strides=2, padding='same',
                              data_format="channels_first")

        convsend3 = tf.layers.conv2d(M8, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consvend3',reuse=reuse)
        convsend3 = tf.layers.batch_normalization(convsend3, training=is_training, axis=1)
        l5 = tf.concat([max3, convsend3],1)

        conve1 = tf.layers.separable_conv2d(l5, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve1', reuse=reuse)
        conve1 = tf.layers.batch_normalization(conve1, training=is_training, axis=1)
        conve2 = tf.layers.separable_conv2d(conve1, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve2', reuse=reuse)
        conev2 = tf.layers.batch_normalization(conve2, training=is_training, axis=1)

        end = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(conev2)

        return end

    def xception_e(self, tensor, name, reuse=None, is_training=True):

        #BEGIN ENTRY FLOW

        conv = tf.layers.conv2d(tensor, basefiltf, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv1',reuse=reuse)#self.conv3by3(tensor,name+'_C1',2, 32)

        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)


        conv2 = tf.layers.conv2d(conv, basefiltf*2, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv2',reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv3 = tf.layers.separable_conv2d(conv2, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv3',reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv4 = tf.layers.separable_conv2d(conv3, basefiltf*4, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv4',reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)

        max1 = tf.layers.max_pooling2d(conv4, pool_size=3, strides=3,padding='same',
                              data_format="channels_first")

        convs1 = tf.layers.conv2d(conv2, 1, kernel_size=1, strides=3, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv1',reuse=reuse)
        convs1 = tf.layers.batch_normalization(convs1, training=is_training, axis=1) #applying BN before relu here and afterwards since it's still a topic of debate, I will assume minimal impact
        l1 = tf.concat([max1,convs1],1)

        l1_b = tf.nn.relu(l1)

        conv5 = tf.layers.separable_conv2d(l1_b, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv5',reuse=reuse)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training, axis=1)
        conv6 = tf.layers.separable_conv2d(conv5, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv6',reuse=reuse)
        conv6 = tf.layers.batch_normalization(conv6, training=is_training, axis=1)
        max2 = tf.layers.max_pooling2d(conv6, pool_size=3, strides=2,padding='same',
                              data_format="channels_first")

        convs2 = tf.layers.conv2d(l1, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv2',reuse=reuse)
        convs2 = tf.layers.batch_normalization(convs2, training=is_training, axis=1)
        l2 = tf.concat([max2, convs2],1)

        # Can't work for the eyes
        # l2_b = tf.nn.relu(l2)
        #
        # conv7 = tf.layers.separable_conv2d(l2_b, 728, kernel_size=[3, 3], strides=2, padding='same',
        #                         activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
        #                         data_format="channels_first",name=name+'_conv7',reuse=reuse)
        # conv7 = tf.layers.batch_normalization(conv7, training=is_training, axis=1)
        # conv8 = tf.layers.separable_conv2d(conv7, 728, kernel_size=[3, 3], strides=3, padding='same',
        #                         kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
        #                         data_format="channels_first",name=name+'_conv8',reuse=reuse)
        # conv8 = tf.layers.batch_normalization(conv8, training=is_training, axis=1)
        # max3 = tf.layers.max_pooling2d(conv8, kernel_size=4, strides=2,
        #                       data_format="channels_first")
        #
        # convs3 = tf.layers.conv2d(conv7, 1, kernel_size=1, strides=2, padding='valid',
        #                         kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
        #                         data_format="channels_first",name=name+'_consv3',reuse=reuse)
        # convs3 = tf.layers.batch_normalization(convs3, training=is_training, axis=1)
        #
        # l3 = tf.concat([max3, convs3])

        #END ENTRY FLOW

        #BEGIN MIDDLE FLOW

        M1 = self.Middle_flow(l2 ,name+'_M1', reuse=reuse, is_training=is_training)
        M2 = self.Middle_flow(M1, name + '_M2', reuse=reuse, is_training=is_training)
        M3 = self.Middle_flow(M2, name + '_M3', reuse=reuse, is_training=is_training)
        M4 = self.Middle_flow(M3, name + '_M4', reuse=reuse, is_training=is_training)
        M5 = self.Middle_flow(M4, name + '_M5', reuse=reuse, is_training=is_training)
        M6 = self.Middle_flow(M5, name + '_M6', reuse=reuse, is_training=is_training)
        M7 = self.Middle_flow(M6, name + '_M7', reuse=reuse, is_training=is_training)
        M8 = self.Middle_flow(M7, name + '_M8', reuse=reuse, is_training=is_training)

        #END MIDDLE FLOW

        #BEGIN END FLOW

        conve1 = tf.layers.separable_conv2d(M8, basefiltf*12, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve1', reuse=reuse)
        conve1 = tf.layers.batch_normalization(conve1, training=is_training, axis=1)
        conve2 = tf.layers.separable_conv2d(conve1, basefiltf*16, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve2', reuse=reuse)
        conev2 = tf.layers.batch_normalization(conve2, training=is_training, axis=1)

        end = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(conev2)

        return end

    def xception_er(self, tensor, name, reuse=None, is_training=True):

        #BEGIN ENTRY FLOW

        conv = tf.layers.conv2d(tensor, basefilte, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv1',reuse=reuse)#self.conv3by3(tensor,name+'_C1',2, 32)

        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)


        conv2 = tf.layers.conv2d(conv, basefilte*2, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_conv2',reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv3 = tf.layers.separable_conv2d(conv2, basefilte*4, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv3',reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv4 = tf.layers.separable_conv2d(conv3, basefilte*4, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv4',reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training, axis=1)

        max1 = tf.layers.max_pooling2d(conv4, pool_size=3, strides=2,padding='same',
                              data_format="channels_first")

        convs1 = tf.layers.conv2d(conv2, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv1',reuse=reuse)
        convs1 = tf.layers.batch_normalization(convs1, training=is_training, axis=1) #applying BN before relu here and afterwards since it's still a topic of debate, I will assume minimal impact
        l1 = tf.concat([max1,convs1],1)

        l1_b = tf.nn.relu(l1)

        conv5 = tf.layers.separable_conv2d(l1_b, basefilte*6, kernel_size=[3, 3], strides=1, padding='same',
                                activation = tf.nn.relu,
                                data_format="channels_first",name=name+'_conv5',reuse=reuse)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training, axis=1)
        conv6 = tf.layers.separable_conv2d(conv5, basefilte*6, kernel_size=[3, 3], strides=1, padding='same',
                                data_format="channels_first",name=name+'_conv6',reuse=reuse)
        conv6 = tf.layers.batch_normalization(conv6, training=is_training, axis=1)
        max2 = tf.layers.max_pooling2d(conv6, pool_size=3, strides=2,padding='same',
                              data_format="channels_first")

        convs2 = tf.layers.conv2d(l1, 1, kernel_size=1, strides=2, padding='valid',
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first",name=name+'_consv2',reuse=reuse)
        convs2 = tf.layers.batch_normalization(convs2, training=is_training, axis=1)
        l2 = tf.concat([max2, convs2],1)


        #END ENTRY FLOW

        #BEGIN MIDDLE FLOW

        M1 = self.Middle_flow(l2,name+'_M1', reuse=reuse, is_training=is_training)
        M2 = self.Middle_flow(M1, name + '_M2', reuse=reuse, is_training=is_training)
        M3 = self.Middle_flow(M2, name + '_M3', reuse=reuse, is_training=is_training)
        M4 = self.Middle_flow(M3, name + '_M4', reuse=reuse, is_training=is_training)
        M5 = self.Middle_flow(M4, name + '_M5', reuse=reuse, is_training=is_training)
        M6 = self.Middle_flow(M5, name + '_M6', reuse=reuse, is_training=is_training)
        M7 = self.Middle_flow(M6, name + '_M7', reuse=reuse, is_training=is_training)
        M8 = self.Middle_flow(M7, name + '_M8', reuse=reuse, is_training=is_training)

        #END MIDDLE FLOW

        #BEGIN END FLOW

        conve1 = tf.layers.separable_conv2d(M8, basefilte*8, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve1', reuse=reuse)
        conve1 = tf.layers.batch_normalization(conve1, training=is_training, axis=1)
        conve2 = tf.layers.separable_conv2d(conve1, basefilte*12, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conve2', reuse=reuse)
        conev2 = tf.layers.batch_normalization(conve2, training=is_training, axis=1)

        end = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(conev2)

        return end


    def Middle_flow(self, tensor, name, reuse=None, is_training=True):

        m1 = tf.nn.relu(tensor)
        conv1 = tf.layers.separable_conv2d(m1, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conv1', reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training, axis=1)
        conv2 = tf.layers.separable_conv2d(conv1, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                           activation=tf.nn.relu,
                                           data_format="channels_first", name=name + '_conv2', reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training, axis=1)
        conv3 = tf.layers.separable_conv2d(conv2, basefiltf*8, kernel_size=[3, 3], strides=1, padding='same',
                                           data_format="channels_first", name=name + '_conv3', reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training, axis=1)
        conv3 = tf.concat([conv3,tensor],1)
        return conv3

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


    def apply_filter(self,tensor,name='Filt',reuse=None):
        #https://arxiv.org/pdf/1903.07296.pdf
        #check with gaussian blur first aswell
        sob = tf.image.sobel_edges(tensor)
        sob = tf.math.sqrt(tf.reduce_sum(tf.math.pow(sob,2),4))
        sobm = tf.reduce_max(sob,[2,3],keepdims=True)
        sob = sob/sobm
        #conv = self.inception_module(tensor,name+'_pre1',filters=32,reuse=reuse,is_training=self.training)# for preprocessing and filtering of data
        ten = tf.concat([sob, tensor], 1)#conv

        return ten


    def __init__(self, drop=0.3): # Dated dropout usage
        self.f = tf.placeholder(tf.float32, shape=(None, 3, 224, 224))
        self.er = tf.placeholder(tf.float32, shape=(None, 3, 60, 224))
        self.le = tf.placeholder(tf.float32, shape=(None, 3, 60, 90))
        self.re = tf.placeholder(tf.float32, shape=(None, 3, 60, 90))
        self.h = tf.placeholder(tf.float32, shape=(None, 2))
        self.fl = tf.placeholder(tf.float32, shape=(None, 33, 2))
        self.g = tf.placeholder(tf.float32, shape=(None, 2))
        self.drop = drop # Dated dropout usage
        self.training = tf.placeholder(tf.bool)
        self.LR = tf.placeholder(tf.float32)
        learning_rate=self.LR

        #pi = tf.constant(m.pi)

        # However both of these are done by batch normalization to a certain degree + noisy dataset
        # Xavier init. uses mean 0 and variance of 1/Neurons (or Navg, being the average of input and output neurons, of a layer

        self.base_init = tf.truncated_normal_initializer(stddev=0.000001)  # Initialise weights 0.01 at max
        self.reg_init = tf.contrib.layers.l2_regularizer(
            scale=0.000001)  # Initialise regularisation Alexnet 0.01 // 0.001 seems to make more sense


        # Begin of filters
        fa =self.apply_filter(self.f,name='face')
        le =self.apply_filter(self.le,name='E')
        re =self.apply_filter(self.re,name='E',reuse=True)

        # Begin of classification
        f = self.xception_f(fa, 'face', is_training=self.training)
        le = self.xception_e(le, 'eye', is_training=self.training)
        re = self.xception_e(re, 'eye',reuse=True, is_training=self.training)

        re = tf.contrib.layers.flatten(re)
        le = tf.contrib.layers.flatten(le)
        f = tf.contrib.layers.flatten(f)
        hex = tf.contrib.layers.flatten(self.h)

        # Begin of dense layers
        x = tf.layers.dense(re, units=128, kernel_initializer=self.base_init, name='FC-RE1', activation=tf.nn.relu,kernel_regularizer=self.reg_init)
        x = tf.layers.batch_normalization(x, training=self.training, axis=1)

        y = tf.layers.dense(le, units=128, kernel_initializer=self.base_init, name='FC-LE1', activation=tf.nn.relu,kernel_regularizer=self.reg_init)
        y = tf.layers.batch_normalization(y, training=self.training, axis=1)
        hex = tf.layers.dense(hex, units=64, kernel_initializer=self.base_init, name='FC-F2', activation=tf.nn.relu)
        hex = tf.layers.batch_normalization(hex, training=self.training, axis=1)

        z = tf.layers.dense(f, units=256, activation=tf.nn.relu, name='F-FC1',kernel_regularizer=self.reg_init)
        z = tf.layers.batch_normalization(z, training=self.training, axis=1)
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu, name='F-FC2',kernel_regularizer=self.reg_init)
        z = tf.layers.batch_normalization(z, training=self.training, axis=1)
        #z = tf.layers.dense(er, units=64, activation=tf.nn.relu, name='ER-FC1')

        x = tf.concat([x, y, z, hex ], 1)

        x = tf.layers.dense(x, units=128, name='FC1',kernel_regularizer=self.reg_init, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=2, name='FC2')
        self.predictions = x#multiply with factor here in case of normalization


        g = self.g# optionally divide by said factor
        #pitch is y,yaw is x

        self.loss = tf.losses.mean_squared_error(g, x)+tf.losses.get_regularization_loss()
        self.angular_loss = util.gaze.tensorflow_angular_error_from_pitchyaw(self.g, self.predictions)
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.extra_update_ops = tf.get_collection(
            tf.GraphKeys. UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = self.trainer.minimize(self.loss)
